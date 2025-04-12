import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os
from datasets import DATASET
from rvsm_model import calculate_metrics

class BugLocalizationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DNNModel(nn.Module):
    def __init__(self, input_size=5):
        super(DNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_predictions).flatten(), np.array(all_labels)

def prepare_data(df, report_ids, scaler=None):
    features = df[['rVSM_similarity', 'collab_filter', 'classname_similarity', 
                  'bug_recency', 'bug_frequency']].values
    
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    labels = df['match'].values.reshape(-1, 1)
    return features, labels, scaler

def train_and_evaluate_dnn(n_splits=5, epochs=100, batch_size=32, learning_rate=0.001):
    # Load features
    features_file = f'output/features_{DATASET.name}.csv'
    df = pd.read_csv(features_file)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup k-fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    unique_reports = df['report_id'].unique()
    
    # Store results
    all_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_reports), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        # Split data
        train_reports = unique_reports[train_idx]
        test_reports = unique_reports[test_idx]
        
        train_df = df[df['report_id'].isin(train_reports)]
        test_df = df[df['report_id'].isin(test_reports)]
        
        # Prepare data
        X_train, y_train, scaler = prepare_data(train_df, train_reports)
        X_test, y_test, _ = prepare_data(test_df, test_reports, scaler)
        
        # Create data loaders
        train_dataset = BugLocalizationDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = DNNModel().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                # Save checkpoint
                checkpoint_dir = 'checkpoints'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, os.path.join(checkpoint_dir, f'model_fold_{fold}.pt'))
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
        
        # Evaluate on test set
        fold_results = []
        for report_id in test_reports:
            test_group = test_df[test_df['report_id'] == report_id]
            X_test_group, y_test_group, _ = prepare_data(test_group, [report_id], scaler)
            test_dataset = BugLocalizationDataset(X_test_group, y_test_group)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
            
            y_pred, y_true = evaluate_model(model, test_loader, device)
            metrics = calculate_metrics(y_true, y_pred)
            metrics['report_id'] = report_id
            metrics['fold'] = fold
            fold_results.append(metrics)
        
        all_results.extend(fold_results)
    
    # Calculate and print average metrics
    results_df = pd.DataFrame(all_results)
    avg_metrics = results_df.drop(['report_id', 'fold'], axis=1).mean()
    
    print("\nDNN Model Results:")
    print("-" * 50)
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return avg_metrics

if __name__ == '__main__':
    train_and_evaluate_dnn() 