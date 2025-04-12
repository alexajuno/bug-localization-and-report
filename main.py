import os
import pandas as pd
from datasets import DATASET
import feature_extraction
import rvsm_model
import dnn_model

def main():
    # Create necessary directories
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Step 1: Extract features
    print("\n=== Extracting Features ===")
    if not os.path.exists(f'output/features_{DATASET.name}.csv'):
        feature_extraction.main()
    else:
        print("Features already extracted, skipping...")
    
    # Step 2: Evaluate RVSM model
    print("\n=== Evaluating RVSM Model ===")
    rvsm_metrics = rvsm_model.evaluate_rvsm()
    
    # Step 3: Train and evaluate DNN model
    print("\n=== Training and Evaluating DNN Model ===")
    dnn_metrics = dnn_model.train_and_evaluate_dnn()
    
    # Save results
    results = pd.DataFrame({
        'RVSM': rvsm_metrics,
        'DNN': dnn_metrics
    })
    
    results.to_csv(f'output/results_{DATASET.name}.csv')
    print("\n=== Final Results ===")
    print(results)
    
    print(f"\nResults saved to output/results_{DATASET.name}.csv")

if __name__ == '__main__':
    main() 