import numpy as np
import pandas as pd
import torch
import os

# NOTE: in theory no mods needed for this file for uncertainty evaluation

def main(args):
    pred_dict = torch.load(args.path)
    gpt_triple_dict = torch.load(f'data_files/{args.dataset}/gpt_triples.pth')
    k_list = [int(k) for k in args.k_list.split(',')]
    
    # Check if this is a BNN result file
    is_bnn = 'bnn' in os.path.basename(args.path)
    
    metric_dict = dict()
    for k in k_list:
        metric_dict[f'ans_recall@{k}'] = []
        metric_dict[f'shortest_path_triple_recall@{k}'] = []
        metric_dict[f'gpt_triple_recall@{k}'] = []
        
        # Add uncertainty-correlation metrics
        if is_bnn:
            metric_dict[f'uncertainty_correct_ratio@{k}'] = []
    
    # Track overall uncertainty metrics
    if is_bnn:
        metric_dict['mean_uncertainty'] = []
        metric_dict['max_uncertainty'] = []
        metric_dict['top_10_uncertainty'] = []
    
    for sample_id in pred_dict:
        if len(pred_dict[sample_id]['scored_triples']) == 0:
            continue
        
        h_list, r_list, t_list, scores, uncertainties = zip(*pred_dict[sample_id]['scored_triples'])
        
        # Track uncertainty metrics if available
        if is_bnn and 'uncertainty_metrics' in pred_dict[sample_id]:
            unc_metrics = pred_dict[sample_id]['uncertainty_metrics']
            metric_dict['mean_uncertainty'].append(unc_metrics.get('mean_uncertainty', 0))
            metric_dict['max_uncertainty'].append(unc_metrics.get('max_uncertainty', 0))
            metric_dict['top_10_uncertainty'].append(unc_metrics.get('top_10_uncertainty', 0))
        
        a_entity_in_graph = set(pred_dict[sample_id]['a_entity_in_graph'])
        if len(a_entity_in_graph) > 0:
            for k in k_list:
                entities_k = set(h_list[:k] + t_list[:k])
                ans_recall = len(a_entity_in_graph & entities_k) / len(a_entity_in_graph)
                metric_dict[f'ans_recall@{k}'].append(ans_recall)
                
                # Calculate correlation between uncertainty and correctness
                if is_bnn:
                    # Create a mask of correct vs incorrect predictions
                    is_correct = []
                    for i in range(min(k, len(h_list))):
                        entity_pair = {h_list[i], t_list[i]}
                        is_correct.append(1 if entity_pair & a_entity_in_graph else 0)
                    
                    if len(is_correct) > 0:
                        # Calculate uncertainty for correct vs incorrect predictions
                        uncertainty_correct = [uncertainties[i] for i in range(len(is_correct)) if is_correct[i] == 1]
                        uncertainty_incorrect = [uncertainties[i] for i in range(len(is_correct)) if is_correct[i] == 0]
                        
                        # Calculate ratio (higher means uncertainty is higher for incorrect predictions, which is good)
                        if uncertainty_correct and uncertainty_incorrect:
                            ratio = np.mean(uncertainty_incorrect) / (np.mean(uncertainty_correct) + 1e-8)
                            metric_dict[f'uncertainty_correct_ratio@{k}'].append(ratio)
        
        triples = list(zip(h_list, r_list, t_list))
        shortest_path_triples = set(pred_dict[sample_id]['target_relevant_triples'])
        if len(shortest_path_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'shortest_path_triple_recall@{k}'].append(
                    len(shortest_path_triples & triples_k) / len(shortest_path_triples)
                )
        
        gpt_triples = set(gpt_triple_dict.get(sample_id, []))
        if len(gpt_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'gpt_triple_recall@{k}'].append(
                    len(gpt_triples & triples_k) / len(gpt_triples)
                )

    # Calculate final metrics
    for metric, val in metric_dict.items():
        if val:  # Only calculate for non-empty lists
            metric_dict[metric] = np.mean(val)
    
    # Create results table
    table_dict = {
        'K': k_list,
        'ans_recall': [
            round(metric_dict[f'ans_recall@{k}'], 3) for k in k_list
        ],
        'shortest_path_triple_recall': [
            round(metric_dict[f'shortest_path_triple_recall@{k}'], 3) for k in k_list
        ],
        'gpt_triple_recall': [
            round(metric_dict[f'gpt_triple_recall@{k}'], 3) for k in k_list
        ]
    }
    
    # Add uncertainty metrics for BNN results
    if is_bnn:
        # Add uncertainty correlation metric
        table_dict['uncertainty_correct_ratio'] = [
            round(metric_dict.get(f'uncertainty_correct_ratio@{k}', 0), 3) for k in k_list
        ]
        
        # Print overall uncertainty metrics
        print("\nUncertainty Metrics:")
        print(f"Mean Uncertainty: {metric_dict.get('mean_uncertainty', 0):.4f}")
        print(f"Max Uncertainty: {metric_dict.get('max_uncertainty', 0):.4f}")
        print(f"Top-10 Uncertainty: {metric_dict.get('top_10_uncertainty', 0):.4f}")
    
    df = pd.DataFrame(table_dict)
    print("\nEvaluation Results:")
    print(df.to_string(index=False))
    
    # Save results if requested
    if args.save_results:
        # Create results directory
        results_dir = os.path.join(os.path.dirname(args.path), "eval_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate result filename
        result_filename = f"eval_results_{'bnn_' if is_bnn else ''}{os.path.basename(args.path).split('.')[0]}.csv"
        df.to_csv(os.path.join(results_dir, result_filename), index=False)
        
        # Save all metrics to a JSON file
        import json
        with open(os.path.join(results_dir, f"all_metrics_{'bnn_' if is_bnn else ''}{os.path.basename(args.path).split('.')[0]}.json"), 'w') as f:
            json.dump(metric_dict, f, indent=2)
        
        print(f"\nResults saved to {results_dir}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to retrieval result')
    parser.add_argument('--k_list', type=str, default='50,100,200,400',
                        help='Comma-separated list of K values for top-K recall evaluation')
    parser.add_argument('--save_results', action='store_true',
                        help='Save evaluation results to a file')
    args = parser.parse_args()
    
    main(args)
