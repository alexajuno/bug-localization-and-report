import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import DATASET
from preprocessing import Parser, ReportPreprocessing, SrcPreprocessing
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple

class SBERTSimilarity:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize SBERT model and device"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def prepare_bug_report_text(self, bug_report) -> str:
        """Combine summary and description from bug report"""
        summary = ' '.join(bug_report.summary['unstemmed'] if isinstance(bug_report.summary, dict) else bug_report.summary)
        description = ' '.join(bug_report.description['unstemmed'] if isinstance(bug_report.description, dict) else bug_report.description)
        return f"{summary} {description}"
    
    def prepare_source_code_text(self, source_file) -> str:
        """Combine relevant parts of source file"""
        # Combine code content and comments
        content = ' '.join(source_file.all_content['unstemmed'] if isinstance(source_file.all_content, dict) else source_file.all_content)
        comments = ' '.join(source_file.comments['unstemmed'] if isinstance(source_file.comments, dict) else source_file.comments)
        
        # Add class and method names as they're important for context
        class_names = ' '.join(source_file.class_names['unstemmed'] if isinstance(source_file.class_names, dict) else source_file.class_names)
        method_names = ' '.join(source_file.method_names['unstemmed'] if isinstance(source_file.method_names, dict) else source_file.method_names)
        
        return f"{class_names} {method_names} {comments} {content}"
    
    def compute_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute embeddings for a list of texts in batches"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def compute_similarities(self, 
                           bug_reports: Dict, 
                           source_files: Dict,
                           top_k: List[int] = [1, 5, 10]) -> Tuple[List[dict], Dict[int, List[float]]]:
        """
        Compute semantic similarities between bug reports and source files
        Returns:
        - List of similarity records
        - Dictionary of top-k accuracy scores
        """
        similarities = []
        correct_predictions = {k: 0 for k in top_k}
        total_reports = 0
        
        # Process bug reports in batches
        for bug_id, bug_report in tqdm(bug_reports.items(), desc="Processing bug reports"):
            # Prepare bug report text
            bug_text = self.prepare_bug_report_text(bug_report)
            bug_embedding = self.model.encode(bug_text, convert_to_numpy=True)
            
            # Prepare all source files for this bug report
            source_texts = []
            file_paths = []
            for file_path, source_file in source_files.items():
                source_texts.append(self.prepare_source_code_text(source_file))
                file_paths.append(file_path)
            
            # Compute embeddings for all source files
            source_embeddings = self.compute_embeddings(source_texts)
            
            # Calculate similarities
            similarity_scores = cosine_similarity([bug_embedding], source_embeddings)[0]
            
            # Create sorted indices for ranking
            ranked_indices = np.argsort(similarity_scores)[::-1]
            
            # Record similarities and calculate top-k accuracy
            actual_files = set(bug_report.fixed_files)
            total_reports += 1
            
            for rank, idx in enumerate(ranked_indices, 1):
                file_path = file_paths[idx]
                score = similarity_scores[idx]
                
                # Record similarity
                similarities.append({
                    'bug_id': bug_id,
                    'file_path': file_path,
                    'similarity_score': score,
                    'rank': rank,
                    'is_correct': file_path in actual_files
                })
                
                # Update top-k accuracy
                if file_path in actual_files:
                    for k in top_k:
                        if rank <= k:
                            correct_predictions[k] += 1
        
        # Calculate top-k accuracy
        accuracy_scores = {k: correct_predictions[k] / total_reports for k in top_k}
        
        return similarities, accuracy_scores

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Initialize parser and preprocessors
    print("Initializing parser and preprocessors...")
    parser = Parser(DATASET)
    
    # Parse and preprocess bug reports
    print("Processing bug reports...")
    bug_reports = parser.report_parser()
    report_preprocessor = ReportPreprocessing(bug_reports)
    report_preprocessor.preprocess()
    
    # Parse and preprocess source files
    print("Processing source files...")
    source_files = parser.src_parser()
    src_preprocessor = SrcPreprocessing(source_files)
    src_preprocessor.preprocess()
    
    # Initialize SBERT similarity calculator
    print("Initializing SBERT model...")
    sbert_sim = SBERTSimilarity()
    
    # Compute similarities and get top-k accuracy
    print("Computing semantic similarities...")
    similarities, accuracy_scores = sbert_sim.compute_similarities(bug_reports, source_files)
    
    # Save results
    print("Saving results...")
    
    # Save detailed similarities
    similarities_df = pd.DataFrame(similarities)
    similarities_df.to_csv(f'output/sbert_similarities_{DATASET.name}.csv', index=False)
    
    # Save accuracy scores
    accuracy_df = pd.DataFrame([accuracy_scores])
    accuracy_df.to_csv(f'output/sbert_accuracy_{DATASET.name}.csv', index=False)
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    for k, score in accuracy_scores.items():
        print(f"Top-{k} Accuracy: {score:.3f}")

if __name__ == '__main__':
    main() 