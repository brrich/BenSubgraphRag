import os
import torch
import numpy as np
import random
from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever_uncertain_bnn import Retriever
from src.setup import set_seed, prepare_sample
from src.model.text_encoders.gte_large_en import GTELargeEN


def sample_qa_pairs(dataset, num_samples=5, seed=42):
    """Sample random QA pairs from the dataset"""
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]

    # Print samples
    print(f"\n{'='*20} SAMPLED Q/A PAIRS {'='*20}")
    for i, sample in enumerate(samples):
        print(f"\nSample #{i}:")
        print(f"Question: {sample['question']}")
        print(f"Answer entity: {sample['a_entity']}")
        print(f"Question entity: {sample['q_entity']}")

    return indices, samples


def run_inference_on_sample(model, sample, device, num_mc_samples=10, top_k=10):
    """Run BNN inference on a single sample with multiple passes"""
    print("DEBUG: Starting inference on sample")
    sample_batch = collate_retriever([sample])
    print("DEBUG: Sample collated")

    try:
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample_batch)
        print(f"DEBUG: Sample prepared, number of triples: {len(h_id_tensor)}")
    except Exception as e:
        print(f"DEBUG: Error during sample preparation: {str(e)}")
        return None, None, None

    entity_list = sample['text_entity_list'] + sample['non_text_entity_list']
    relation_list = sample['relation_list']
    print(f"DEBUG: Entity list length: {len(entity_list)}, Relation list length: {len(relation_list)}")

    if len(h_id_tensor) == 0:
        print("No triples found for this sample")
        return None, None, None

    # Perform multiple forward passes with Bayesian sampling
    print("DEBUG: Starting Monte Carlo sampling")
    all_pred_logits = []
    try:
        for i in tqdm(range(num_mc_samples), desc="Running MC samples"):
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            all_pred_logits.append(pred_triple_logits.reshape(-1))
            print(f"DEBUG: Completed MC sample {i+1}/{num_mc_samples}")
    except Exception as e:
        print(f"DEBUG: Error during model forward pass: {str(e)}")
        return None, None, None

    print("DEBUG: Monte Carlo sampling completed")

    try:
        # Stack predictions from all passes [mc_samples, num_triples]
        stacked_preds = torch.stack(all_pred_logits)
        print(f"DEBUG: Stacked predictions shape: {stacked_preds.shape}")

        # Calculate mean and std across MC samples
        mean_logits = torch.mean(stacked_preds, dim=0)
        std_logits = torch.std(stacked_preds, dim=0)
        print(f"DEBUG: Mean and std calculated")

        # Convert logits to probabilities
        pred_triple_scores = torch.sigmoid(mean_logits)
        pred_triple_uncertainties = std_logits
        print(f"DEBUG: Converted to probabilities")

        # Get top K triples based on mean scores
        top_k_results = torch.topk(pred_triple_scores, min(top_k, len(pred_triple_scores)))
        top_k_scores = top_k_results.values.cpu().tolist()
        top_k_triple_ids = top_k_results.indices.cpu().tolist()
        print(f"DEBUG: Top {len(top_k_scores)} triples selected")

        # Get corresponding uncertainty values
        top_k_uncertainties = pred_triple_uncertainties[top_k_triple_ids].cpu().tolist()

        # Format triples for printing
        top_k_triples = []
        for j, triple_id in enumerate(top_k_triple_ids):
            h_id = h_id_tensor[triple_id].item()
            r_id = r_id_tensor[triple_id].item()
            t_id = t_id_tensor[triple_id].item()

            top_k_triples.append({
                'head': entity_list[h_id],
                'relation': relation_list[r_id],
                'tail': entity_list[t_id],
                'score': top_k_scores[j],
                'uncertainty': top_k_uncertainties[j]
            })

        print(f"DEBUG: Formatted {len(top_k_triples)} triples for return")
        avg_uncertainty = np.mean(top_k_uncertainties)

        return top_k_triples, avg_uncertainty, std_logits
    except Exception as e:
        print(f"DEBUG: Error during post-processing: {str(e)}")
        return None, None, None


def find_question_entity_for_target(test_set, target_entity):
    """Find a question entity related to the target entity"""
    for sample in test_set.processed_dict_list:
        if target_entity in sample['a_entity']:
            return sample['q_entity'][0] if sample['q_entity'] else None
    return None


def get_samples_with_entity(test_set, entity_name, as_question=True):
    """Find samples that have a specific entity as either question or answer entity"""
    matching_samples = []

    for sample in test_set.processed_dict_list:
        if as_question and entity_name in sample['q_entity']:
            matching_samples.append(sample)
        elif not as_question and entity_name in sample['a_entity']:
            matching_samples.append(sample)

    return matching_samples


def create_custom_sample_from_existing(base_sample, custom_question, target_entity=None):
    """Create a custom sample from an existing one, changing the question but keeping the graph"""
    custom_sample = base_sample.copy()
    custom_sample['question'] = custom_question

    if target_entity:
        custom_sample['a_entity'] = [target_entity]

    return custom_sample


def generate_query_embedding(text_encoder, query_text, device):
    """Generate embedding for a custom query text"""
    return text_encoder.embed([query_text]).to(device)


def run_custom_query_analysis(model, text_encoder, test_set, custom_query, target_entity, device, num_mc_samples=10, top_k=10):
    """Run analysis on a custom query and compare with test set results"""
    print(f"\n{'='*20} CUSTOM QUERY ANALYSIS {'='*20}")
    print(f"Custom Query: {custom_query}")
    print(f"Target Entity: {target_entity}")

    # Find a suitable question entity for the target
    q_entity = find_question_entity_for_target(test_set, target_entity)
    if not q_entity:
        print(f"Could not find a question entity related to {target_entity}")
        return None

    print(f"Using question entity: {q_entity}")

    # Find samples with the same question entity for comparison
    comparison_samples = get_samples_with_entity(test_set, q_entity, as_question=True)
    print(f"Found {len(comparison_samples)} test samples with question entity {q_entity}")

    if len(comparison_samples) == 0:
        print("No comparison samples found")
        return None

    # Use the first comparison sample as a base
    base_sample = comparison_samples[0]
    custom_sample = create_custom_sample_from_existing(base_sample, custom_query, target_entity)

    # Generate embeddings for the custom query
    custom_q_emb = generate_query_embedding(text_encoder, custom_query, device)

    # Replace the embedding in the custom sample
    custom_sample['q_emb'] = custom_q_emb

    # Run inference on custom query
    print(f"\n{'='*20} RUNNING INFERENCE ON CUSTOM QUERY {'='*20}")
    custom_triples, custom_avg_uncertainty, custom_uncertainties = run_inference_on_sample(
        model, custom_sample, device, num_mc_samples, top_k)

    if not custom_triples:
        print("No results for custom query")
        return None

    # Run inference on comparison samples
    comparison_uncertainties = []
    for i, sample in enumerate(comparison_samples):
        print(f"\n{'='*20} COMPARISON SAMPLE #{i+1} {'='*20}")
        print(f"Question: {sample['question']}")
        print(f"Answer entity: {sample['a_entity']}")

        triples, avg_uncertainty, uncertainties = run_inference_on_sample(
            model, sample, device, num_mc_samples, top_k)

        if triples:
            comparison_uncertainties.append(avg_uncertainty)

    # Calculate average uncertainty for comparison samples
    if comparison_uncertainties:
        avg_comparison_uncertainty = np.mean(comparison_uncertainties)
        std_comparison_uncertainty = np.std(comparison_uncertainties)

        # Print results
        print(f"\n{'='*20} UNCERTAINTY COMPARISON {'='*20}")
        print(f"Custom query average uncertainty: {custom_avg_uncertainty:.4f}")
        print(f"Test set average uncertainty: {avg_comparison_uncertainty:.4f} (std: {std_comparison_uncertainty:.4f})")
        print(f"Relative difference: {(custom_avg_uncertainty - avg_comparison_uncertainty) / avg_comparison_uncertainty * 100:.2f}%")

        return custom_triples, custom_avg_uncertainty, comparison_uncertainties
    else:
        print("No valid comparison samples")
        return None


def main(args):
    # Set device
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Load model checkpoint
        print(f"Loading model from {args.model_path}")
        cpt = torch.load(args.model_path, map_location='cpu')
        config = cpt['config']
        print("DEBUG: Model checkpoint loaded successfully")

        # Set seed for reproducibility
        set_seed(config['env']['seed'])
        torch.set_num_threads(config['env']['num_threads'])

        # Load test dataset
        print("Loading test dataset")
        test_set = RetrieverDataset(
            config=config, split='test', skip_no_path=False)
        print(f"DEBUG: Loaded test dataset with {len(test_set)} samples")

        # Sample QA pairs
        sample_indices, samples = sample_qa_pairs(test_set, args.num_samples, args.seed)
        print(f"DEBUG: Sampled {len(samples)} QA pairs")

        # Initialize model
        emb_size = test_set[0]['q_emb'].shape[-1]
        model = Retriever(emb_size, **config['retriever']).to(device)
        model.load_state_dict(cpt['model_state_dict'])
        model.eval()  # For BNN, this doesn't affect Bayesian sampling
        print("DEBUG: Model initialized and loaded state dict")

        # Custom query mode
        if args.custom_query:
            # Initialize text encoder
            text_encoder = GTELargeEN(device=device)

            # Run custom query analysis
            run_custom_query_analysis(
                model,
                text_encoder,
                test_set,
                args.custom_query,
                args.target_entity,
                device,
                args.mc_samples,
                args.top_k
            )
        # Original mode
        else:
            # Run inference on selected sample
            if args.sample_id >= 0 and args.sample_id < len(sample_indices):
                selected_idx = sample_indices[args.sample_id]
                selected_sample = test_set[selected_idx]

                print(f"\n{'='*20} RUNNING INFERENCE ON SAMPLE #{args.sample_id} {'='*20}")
                print(f"Question: {selected_sample['question']}")
                print(f"Answer entity: {selected_sample['a_entity']}")

                # Run inference
                print("DEBUG: Starting run_inference_on_sample")
                top_k_triples, avg_uncertainty, all_uncertainties = run_inference_on_sample(
                    model, selected_sample, device, args.mc_samples, args.top_k)
                print("DEBUG: Completed run_inference_on_sample")

                print(f"\n{'='*20} TOP {args.top_k} TRIPLES {'='*20}")

                if top_k_triples:
                    # Print results
                    for i, triple in enumerate(top_k_triples):
                        print(f"{i+1}. {triple['head']} --- {triple['relation']} --> {triple['tail']}")
                        print(f"   Score: {triple['score']:.4f}, Uncertainty: {triple['uncertainty']:.4f}")

                    print(f"\n{'='*20} UNCERTAINTY METRICS {'='*20}")
                    print(f"Average uncertainty across top {args.top_k} triples: {avg_uncertainty:.4f}")
                    print(f"Overall mean uncertainty: {all_uncertainties.mean().item():.4f}")
                    print(f"Overall max uncertainty: {all_uncertainties.max().item():.4f}")
                else:
                    print("No triples found for this sample or an error occurred during processing")
            else:
                print(f"Invalid sample ID. Please choose from 0 to {len(sample_indices)-1}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to a saved model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of QA pairs to sample from test set')
    parser.add_argument('--sample_id', type=int, default=0,
                        help='ID of the sample to run inference on (from the sampled set)')
    parser.add_argument('--mc_samples', type=int, default=10,
                        help='Number of Monte Carlo passes for BNN inference')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top triples to display')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    # Add custom query arguments
    parser.add_argument('--custom_query', type=str, default=None,
                        help='Custom query text for uncertainty analysis')
    parser.add_argument('--target_entity', type=str, default=None,
                        help='Target entity for the custom query')

    args = parser.parse_args()

    main(args)
