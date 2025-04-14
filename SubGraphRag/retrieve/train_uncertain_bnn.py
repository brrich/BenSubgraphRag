import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb
import json

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever_uncertain_bnn import Retriever
from src.setup import set_seed, prepare_sample

@torch.no_grad()
def eval_epoch(config, device, data_loader, model, mc_samples=10):
    model.eval()  # Doesn't influence Bayesian sampling in the BNN model
    
    metric_dict = defaultdict(list)
    uncertainty_dict = defaultdict(list)
    
    for sample in tqdm(data_loader):
        # Prepare sample once
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
        
        # Run K forward passes for Monte Carlo estimation of predictive distribution
        all_pred_logits = []
        for _ in range(mc_samples):
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot).reshape(-1)
            all_pred_logits.append(pred_triple_logits)
        
        # Stack predictions from all passes [mc_samples, num_triples]
        stacked_preds = torch.stack(all_pred_logits)
        
        # Calculate mean and std across MC samples
        mean_logits = torch.mean(stacked_preds, dim=0)
        std_logits = torch.std(stacked_preds, dim=0)
        
        # Store uncertainty metrics
        uncertainty_dict['mean_std'].append(std_logits.mean().item())
        uncertainty_dict['max_std'].append(std_logits.max().item())
        
        # Use mean predictions for ranking (like in original retriever)
        sorted_triple_ids_pred = torch.argsort(mean_logits, descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(len(triple_ranks_pred))
        
        target_triple_ids = target_triple_probs.nonzero().squeeze(-1)
        num_target_triples = len(target_triple_ids)
        
        if num_target_triples == 0:
            continue

        # Calculate correlation between uncertainty and correctness
        correctness = torch.zeros_like(mean_logits)
        correctness[target_triple_ids] = 1.0
        if len(std_logits) > 1:  # Need at least 2 samples for correlation
            corr = torch.corrcoef(torch.stack([std_logits.cpu(), correctness.cpu()]))[0, 1].item()
            uncertainty_dict['uncertainty_correctness_corr'].append(corr)
            
        num_total_entities = len(entity_embs) + num_non_text_entities
        for k in config['eval']['k_list']:
            # Same metrics as in original retriever.py
            recall_k_sample = (
                triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(
                recall_k_sample / num_target_triples)
            
            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
            recall_k_sample_ans = entity_mask_k[a_entity_id_list].sum().item()
            metric_dict[f'ans_recall@{k}'].append(
                recall_k_sample_ans / len(a_entity_id_list))
            
            # Add uncertainty metrics for top-k predictions
            if k <= 100:  # Only compute for smaller k values to save computation
                top_k_uncertainty = std_logits[sorted_triple_ids_pred[:k]].mean().item()
                uncertainty_dict[f'top{k}_uncertainty'] = top_k_uncertainty

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)
    
    # Process uncertainty metrics
    for key, val in uncertainty_dict.items():
        if isinstance(val, list) and val:
            uncertainty_dict[key] = np.mean(val)
    
    return metric_dict, uncertainty_dict

def train_epoch(device, train_loader, model, optimizer, kl_weight=1.0):
    model.train()
    epoch_loss = 0
    epoch_kl_loss = 0
    epoch_nll_loss = 0
    num_batches = 0
    
    for sample in tqdm(train_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
        
        # Calculate the negative log likelihood (data loss)
        nll_loss = F.binary_cross_entropy_with_logits(
            pred_triple_logits, target_triple_probs)
        
        # Get the KL divergence from the model
        kl_div = model.get_kl_divergence()
        
        # ELBO loss = NLL + KL weight * KL divergence
        # KL weight can be annealed to improve training
        loss = nll_loss + kl_weight * kl_div
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss components
        epoch_loss += loss.item()
        epoch_nll_loss += nll_loss.item()
        epoch_kl_loss += kl_div.item() * kl_weight
        num_batches += 1
    
    epoch_loss /= num_batches
    epoch_nll_loss /= num_batches
    epoch_kl_loss /= num_batches
    
    log_dict = {
        'loss': epoch_loss,
        'nll_loss': epoch_nll_loss,
        'kl_loss': epoch_kl_loss
    }
    return log_dict

def main(args):
    # Set wandb to offline mode if no internet connection
    os.environ["WANDB_MODE"] = "offline"
    
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    # Update number of epochs if specified in args
    if args.num_epochs:
        config['train']['num_epochs'] = args.num_epochs
    
    device = torch.device('cuda:0')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']
    exp_name = f'{exp_prefix}_{ts}'
    
    # Initialize wandb (will work in offline mode if no internet)
    wandb.init(
        project=f'{args.dataset}',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0]
    )
    
    os.makedirs(exp_name, exist_ok=True)
    
    # Create a log file for metrics (separate from wandb)
    log_file = os.path.join(exp_name, 'metrics.json')

    train_set = RetrieverDataset(config=config, split='train')
    val_set = RetrieverDataset(config=config, split='val')
    
    # Load test set if available
    try:
        test_set = RetrieverDataset(config=config, split='test')
        test_loader = DataLoader(
            test_set, batch_size=1, collate_fn=collate_retriever)
        has_test_set = True
        print(f"Test set loaded with {len(test_set)} samples")
    except Exception as e:
        print(f"Warning: Could not load test set: {str(e)}")
        has_test_set = False

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(
        val_set, batch_size=1, collate_fn=collate_retriever)
    
    emb_size = train_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])

    num_patient_epochs = 0
    best_val_metric = 0
    
    # Dictionary to track metrics over time
    all_metrics = {
        'train': [],
        'val': [],
        'test': [],
        'uncertainty': [],
        'epoch_history': {
            'train_loss': [],
            'train_nll_loss': [],
            'train_kl_loss': [],
            'val_recall': [],
            'test_recall': [] if has_test_set else None
        }
    }
    
    # KL annealing factor - gradually increase to prevent mode collapse
    # Start with a small factor and increase over time
    kl_weight = args.kl_weight_start
    kl_weight_step = (args.kl_weight_end - args.kl_weight_start) / config['train']['num_epochs']
    
    for epoch in range(config['train']['num_epochs']):
        num_patient_epochs += 1
        
        # Evaluate on validation set using MC sampling from the Bayesian posterior
        val_eval_dict, val_uncertainty_dict = eval_epoch(config, device, val_loader, model, args.mc_samples)
        target_val_metric = val_eval_dict['triple_recall@100']
        
        # Evaluate on test set if available
        if has_test_set:
            print(f"Evaluating on test set (epoch {epoch})...")
            test_eval_dict, test_uncertainty_dict = eval_epoch(config, device, test_loader, model, args.mc_samples)
            all_metrics['test'].append({
                'epoch': epoch,
                **test_eval_dict
            })
            # Track test recall for summary
            all_metrics['epoch_history']['test_recall'].append({
                'epoch': epoch,
                'triple_recall@100': test_eval_dict['triple_recall@100'],
                'ans_recall@100': test_eval_dict['ans_recall@100']
            })
        
        # Save metrics
        all_metrics['val'].append({
            'epoch': epoch,
            **val_eval_dict
        })
        all_metrics['uncertainty'].append({
            'epoch': epoch,
            **val_uncertainty_dict
        })
        
        # Track validation recall for summary
        all_metrics['epoch_history']['val_recall'].append({
            'epoch': epoch, 
            'triple_recall@100': val_eval_dict['triple_recall@100'],
            'ans_recall@100': val_eval_dict['ans_recall@100']
        })
        
        if target_val_metric > best_val_metric:
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict()
            }
            best_epoch = epoch
            torch.save(best_state_dict, os.path.join(exp_name, f'cpt.pth'))

            # Log to wandb
            val_log = {'val/epoch': epoch}
            for key, val in val_eval_dict.items():
                val_log[f'val/{key}'] = val
            for key, val in val_uncertainty_dict.items():
                val_log[f'val/uncertainty_{key}'] = val
            wandb.log(val_log, step=epoch)
            
            # Also save test metrics at best validation point
            if has_test_set:
                best_test_metrics = test_eval_dict
                best_test_uncertainty = test_uncertainty_dict
            
            # Save the current best metrics
            with open(os.path.join(exp_name, 'best_metrics.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'metrics': val_eval_dict,
                    'uncertainty': val_uncertainty_dict,
                    'test_metrics': test_eval_dict if has_test_set else None
                }, f, indent=2)

        # Train for one epoch with the current KL weight
        train_log_dict = train_epoch(device, train_loader, model, optimizer, kl_weight=kl_weight)
        train_log_dict.update({
            'num_patient_epochs': num_patient_epochs,
            'epoch': epoch,
            'kl_weight': kl_weight
        })
        
        # Save training metrics
        all_metrics['train'].append({
            'epoch': epoch,
            **train_log_dict
        })
        
        # Track training loss components for summary
        all_metrics['epoch_history']['train_loss'].append({
            'epoch': epoch,
            'loss': train_log_dict['loss']
        })
        all_metrics['epoch_history']['train_nll_loss'].append({
            'epoch': epoch,
            'nll_loss': train_log_dict['nll_loss']
        })
        all_metrics['epoch_history']['train_kl_loss'].append({
            'epoch': epoch,
            'kl_loss': train_log_dict['kl_loss']
        })
        
        # Log to wandb
        wandb.log(train_log_dict, step=epoch)
        
        # Save all metrics to file
        with open(log_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
            
        # Print progress
        print(f"Epoch {epoch}: train_loss={train_log_dict['loss']:.4f}, val_recall@100={val_eval_dict['triple_recall@100']:.4f}, kl_weight={kl_weight:.6f}")
        if has_test_set:
            print(f"           test_recall@100={test_eval_dict['triple_recall@100']:.4f}")
            
        # Increase KL weight according to annealing schedule
        kl_weight = min(args.kl_weight_end, kl_weight + kl_weight_step)
            
        if num_patient_epochs == config['train']['patience']:
            print(f"Early stopping triggered after {num_patient_epochs} epochs without improvement")
            break
            
    # Save final metrics summary with epoch-by-epoch history
    summary_file = os.path.join(exp_name, 'summary.json')
    summary = {
        'best_epoch': best_epoch,
        'num_epochs_trained': epoch + 1,
        'best_triple_recall@100': best_val_metric,
        'mc_samples': args.mc_samples,
        'kl_weight_final': kl_weight,
        'kl_weight_start': args.kl_weight_start,
        'kl_weight_end': args.kl_weight_end,
        'final_metrics': {
            'val': val_eval_dict,
            'test': test_eval_dict if has_test_set else None
        },
        'best_metrics': {
            'val': all_metrics['val'][best_epoch],
            'test': best_test_metrics if has_test_set else None
        },
        'uncertainty': val_uncertainty_dict,
        'epoch_history': all_metrics['epoch_history']
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Training complete. Best validation triple_recall@100: {best_val_metric:.4f}")
    if has_test_set:
        print(f"Best test triple_recall@100: {best_test_metrics['triple_recall@100']:.4f}")
    print(f"Results saved to {exp_name}")
    print(f"Uncertainty metrics: {json.dumps(val_uncertainty_dict, indent=2)}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('--mc_samples', type=int, default=15, 
                       help='Number of Monte Carlo samples for uncertainty estimation (default: 100)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--kl_weight_start', type=float, default=0.0001,
                       help='Initial KL divergence weight for ELBO loss (default: 0.0001)')
    parser.add_argument('--kl_weight_end', type=float, default=0.01,
                       help='Final KL divergence weight for ELBO loss (default: 0.01)')
    args = parser.parse_args()
    
    main(args)
