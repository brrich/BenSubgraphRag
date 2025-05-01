import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever_uncertain_bnn import Retriever
from src.setup import set_seed, prepare_sample


def sample_dataset(dataset, num_samples=100, seed=42):
    """Sample random samples from the dataset"""
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    return samples


def run_inference_on_sample(model, sample, device, num_mc_samples=10, top_k=10, verbose=False):
    """Run BNN inference on a single sample with multiple passes"""
    if verbose:
        print("Starting inference on sample")

    sample_batch = collate_retriever([sample])

    try:
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample_batch)

        if verbose:
            print(f"Sample prepared, number of triples: {len(h_id_tensor)}")
    except Exception as e:
        if verbose:
            print(f"Error during sample preparation: {str(e)}")
        return None, None, None

    if len(h_id_tensor) == 0:
        if verbose:
            print("No triples found for this sample")
        return None, None, None

    # Perform multiple forward passes with Bayesian sampling
    all_pred_logits = []
    try:
        for i in range(num_mc_samples):
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            all_pred_logits.append(pred_triple_logits.reshape(-1))
    except Exception as e:
        if verbose:
            print(f"Error during model forward pass: {str(e)}")
        return None, None, None

    try:
        # Stack predictions from all passes [mc_samples, num_triples]
        stacked_preds = torch.stack(all_pred_logits)

        # Calculate mean and std across MC samples
        mean_logits = torch.mean(stacked_preds, dim=0)
        std_logits = torch.std(stacked_preds, dim=0)

        # Convert logits to probabilities
        pred_triple_scores = torch.sigmoid(mean_logits)
        pred_triple_uncertainties = std_logits

        # Get top K triples based on mean scores
        top_k_results = torch.topk(pred_triple_scores, min(top_k, len(pred_triple_scores)))
        top_k_scores = top_k_results.values.cpu().tolist()
        top_k_triple_ids = top_k_results.indices.cpu().tolist()

        # Get corresponding uncertainty values
        top_k_uncertainties = pred_triple_uncertainties[top_k_triple_ids].cpu().tolist()

        avg_uncertainty = np.mean(top_k_uncertainties)

        return top_k_scores, avg_uncertainty, std_logits
    except Exception as e:
        if verbose:
            print(f"Error during post-processing: {str(e)}")
        return None, None, None


def analyze_uncertainty_distribution(model_path, dataset_name, mc_samples=10, num_samples=100, top_k=10, seed=42):
    """
    Analyze the uncertainty distribution across training, validation, and test sets

    Args:
        model_path: Path to the model checkpoint (.pth file)
        dataset_name: Name of the dataset ('webqsp' or 'cwq')
        mc_samples: Number of Monte Carlo forward passes for BNN
        num_samples: Number of samples to use from each split
        top_k: Number of top triples to consider
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing uncertainty statistics for each split
    """
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(seed)

    try:
        # Load model checkpoint
        print(f"Loading model from {model_path}")
        cpt = torch.load(model_path, map_location='cpu')
        config = cpt['config']

        # Make sure the dataset matches
        if 'dataset' in config and config['dataset']['name'] != dataset_name:
            config['dataset']['name'] = dataset_name
            print(f"Warning: Changed dataset in config from {config['dataset']['name']} to {dataset_name}")

        # Initialize model
        print("Initializing model...")

        # Load dataset splits
        print("Loading datasets...")
        train_set = RetrieverDataset(config=config, split='train', skip_no_path=False)
        valid_set = RetrieverDataset(config=config, split='valid', skip_no_path=False)
        test_set = RetrieverDataset(config=config, split='test', skip_no_path=False)

        print(f"Loaded datasets - Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

        # Sample from each split
        print(f"Sampling {num_samples} examples from each split...")
        train_samples = sample_dataset(train_set, num_samples, seed)
        valid_samples = sample_dataset(valid_set, num_samples, seed)
        test_samples = sample_dataset(test_set, num_samples, seed)

        # Initialize model
        emb_size = train_set[0]['q_emb'].shape[-1]
        model = Retriever(emb_size, **config['retriever']).to(device)
        model.load_state_dict(cpt['model_state_dict'])
        model.eval()  # For BNN, this doesn't affect Bayesian sampling

        # Collect uncertainties for each split
        splits = {
            'train': train_samples,
            'valid': valid_samples,
            'test': test_samples
        }

        results = {}

        for split_name, samples in splits.items():
            print(f"\nProcessing {split_name} split...")
            uncertainties = []
            mean_scores = []

            for i, sample in enumerate(tqdm(samples, desc=f"{split_name}")):
                scores, avg_uncertainty, all_uncertainties = run_inference_on_sample(
                    model, sample, device, mc_samples, top_k, verbose=False)

                if scores is not None and avg_uncertainty is not None:
                    uncertainties.append(avg_uncertainty)
                    mean_scores.append(np.mean(scores))

            results[split_name] = {
                'uncertainties': uncertainties,
                'mean_scores': mean_scores,
                'mean_uncertainty': np.mean(uncertainties) if uncertainties else None,
                'std_uncertainty': np.std(uncertainties) if uncertainties else None,
                'mean_score': np.mean(mean_scores) if mean_scores else None,
                'std_score': np.std(mean_scores) if mean_scores else None
            }

            print(f"{split_name} - Mean uncertainty: {results[split_name]['mean_uncertainty']:.4f}, "
                 f"Std: {results[split_name]['std_uncertainty']:.4f}")
            print(f"{split_name} - Mean score: {results[split_name]['mean_score']:.4f}, "
                 f"Std: {results[split_name]['std_score']:.4f}")

        return results

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        return None


def plot_uncertainty_distribution(results, output_path=None):
    """
    Plot the distribution of uncertainty values for each split

    Args:
        results: Dictionary containing uncertainty statistics for each split
        output_path: Path to save the plot (if None, plot is shown)
    """
    if not results:
        print("No results to plot")
        return

    plt.figure(figsize=(10, 6))

    colors = {'train': 'blue', 'valid': 'green', 'test': 'red'}
    bins = 20

    for split_name, color in colors.items():
        if split_name in results and results[split_name]['uncertainties']:
            uncertainties = results[split_name]['uncertainties']
            plt.hist(uncertainties, bins=bins, alpha=0.5, color=color, label=f'{split_name} (n={len(uncertainties)})')

    plt.xlabel('Uncertainty')
    plt.ylabel('Count')
    plt.title('Distribution of BNN Uncertainty Across Dataset Splits')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def main(args):
    # Run uncertainty distribution analysis
    results = analyze_uncertainty_distribution(
        model_path=args.model_path,
        dataset_name=args.dataset,
        mc_samples=args.mc_samples,
        num_samples=args.num_samples,
        top_k=args.top_k,
        seed=args.seed
    )

    if results:
        # Plot the distributions
        output_file = f"uncertainty_dist_{args.dataset}_{args.num_samples}samples_{args.mc_samples}mc.png"
        plot_uncertainty_distribution(results, output_path=output_file)

        # Print summary stats
        print("\n===== SUMMARY STATISTICS =====")
        for split_name in results:
            if results[split_name]['uncertainties']:
                print(f"{split_name.upper()} SPLIT:")
                print(f"  Mean uncertainty: {results[split_name]['mean_uncertainty']:.4f} ± {results[split_name]['std_uncertainty']:.4f}")
                print(f"  Mean score: {results[split_name]['mean_score']:.4f} ± {results[split_name]['std_score']:.4f}")
                print(f"  Number of valid samples: {len(results[split_name]['uncertainties'])}/{args.num_samples}")
                print()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to a saved model checkpoint (.pth file)')
    parser.add_argument('--dataset', type=str, required=True, choices=['webqsp', 'cwq'],
                        help='Name of the dataset to use (webqsp or cwq)')
    parser.add_argument('--mc_samples', type=int, default=10,
                        help='Number of Monte Carlo passes for BNN inference')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to use from each split')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top triples to consider')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    main(args)
