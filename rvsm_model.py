import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from datasets import DATASET

def calculate_metrics(y_true, y_pred, k_values=[1, 5, 10]):
    """Calculate Top-k accuracy, MAP and MRR"""
    metrics = {}
    
    # Calculate Top-k accuracy
    for k in k_values:
        top_k_pred = np.argsort(y_pred)[-k:][::-1]
        top_k_accuracy = np.mean([1 if np.any(y_true[top_k_pred] == 1) else 0])
        metrics[f'top_{k}_accuracy'] = top_k_accuracy
    
    # Calculate MAP
    metrics['map'] = average_precision_score(y_true, y_pred)
    
    # Calculate MRR
    ranked_indices = np.argsort(y_pred)[::-1]
    for i, idx in enumerate(ranked_indices, 1):
        if y_true[idx] == 1:
            metrics['mrr'] = 1.0 / i
            break
    else:
        metrics['mrr'] = 0.0
    
    return metrics

def evaluate_rvsm():
    """Evaluate RVSM model using only rVSM similarity scores"""
    # Load features
    features_file = f'output/features_{DATASET.name}.csv'
    df = pd.read_csv(features_file)
    
    # Group by report_id to evaluate each bug report separately
    results = []
    for report_id, group in df.groupby('report_id'):
        y_true = group['match'].values
        y_pred = group['rVSM_similarity'].values
        
        metrics = calculate_metrics(y_true, y_pred)
        metrics['report_id'] = report_id
        results.append(metrics)
    
    # Calculate average metrics
    results_df = pd.DataFrame(results)
    avg_metrics = results_df.drop('report_id', axis=1).mean()
    
    print("\nRVSM Model Results:")
    print("-" * 50)
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return avg_metrics

if __name__ == '__main__':
    evaluate_rvsm() 