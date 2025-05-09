import os
import torch
import wandb
import numpy as np

from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever_uncertain_bnn import Retriever  # Import BNN version
from src.setup import set_seed, prepare_sample

@torch.no_grad()
def main(args):
    # Set wandb to offline mode if no internet connection
    os.environ["WANDB_MODE"] = "offline"
    
    device = torch.device(f'cuda:0')
    
    cpt = torch.load(args.path, map_location='cpu')
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    
    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()  # For the BNN model, this doesn't affect Bayesian sampling
    
    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            # Perform multiple forward passes with Bayesian sampling
            all_pred_logits = []
            for _ in range(args.mc_samples):
                pred_triple_logits = model(
                    h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                    num_non_text_entities, relation_embs, topic_entity_one_hot)
                all_pred_logits.append(pred_triple_logits.reshape(-1))
            
            # Stack predictions from all passes [mc_samples, num_triples]
            stacked_preds = torch.stack(all_pred_logits)
            
            # Calculate mean and std across MC samples
            mean_logits = torch.mean(stacked_preds, dim=0)
            std_logits = torch.std(stacked_preds, dim=0)
            
            # Convert logits to probabilities
            pred_triple_scores = torch.sigmoid(mean_logits)
            pred_triple_uncertainties = std_logits  # Keep as is for uncertainty
            
            # Get top K triples based on mean scores
            top_K_results = torch.topk(pred_triple_scores, 
                                       min(args.max_K, len(pred_triple_scores)))
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()
            
            # Get corresponding uncertainty values for top K triples
            top_K_uncertainties = pred_triple_uncertainties[top_K_triple_IDs].cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j],
                    top_K_uncertainties[j]  # Add uncertainty metric
                ))

            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        # Calculate overall uncertainty metrics
        uncertainty_metrics = {}
        if len(h_id_tensor) != 0:
            uncertainty_metrics = {
                'mean_uncertainty': std_logits.mean().item(),
                'max_uncertainty': std_logits.max().item(),
                'top_10_uncertainty': std_logits[top_K_triple_IDs[:min(10, len(top_K_triple_IDs))]].mean().item() if len(top_K_triple_IDs) > 0 else 0
            }

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples,
            'uncertainty_metrics': uncertainty_metrics
        }
        
        pred_dict[raw_sample['id']] = sample_dict

    root_path = os.path.dirname(args.path)
    torch.save(pred_dict, os.path.join(root_path, f'retrieval_result_bnn_mc{args.mc_samples}.pth'))
    
    # Also save a summary of uncertainty metrics
    uncertainty_summary = {
        'sample_uncertainties': {id: data['uncertainty_metrics'] for id, data in pred_dict.items() if 'uncertainty_metrics' in data},
        'overall_mean_uncertainty': np.mean([data['uncertainty_metrics']['mean_uncertainty'] 
                                           for data in pred_dict.values() 
                                           if 'uncertainty_metrics' in data and 'mean_uncertainty' in data['uncertainty_metrics']])
    }
    torch.save(uncertainty_summary, os.path.join(root_path, f'uncertainty_summary_bnn_mc{args.mc_samples}.pth'))

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    parser.add_argument('--mc_samples', type=int, default=100,
                        help='Number of Monte Carlo Bayesian forward passes (default: 20)')
    args = parser.parse_args()
    
    main(args)
