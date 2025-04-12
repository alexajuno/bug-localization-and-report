import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from collections import defaultdict
from datetime import datetime
from datasets import DATASET
from preprocessing import Parser, ReportPreprocessing, SrcPreprocessing

def extract_text_features(text_tokens):
    """Convert tokenized text back to string for TF-IDF"""
    if isinstance(text_tokens, dict):
        # Handle stemmed/unstemmed dict format
        return ' '.join(text_tokens['stemmed'])
    return ' '.join(text_tokens)

def calculate_rvsm_similarity(bug_report, source_file):
    """Calculate rVSM (revised Vector Space Model) similarity between bug report and source file"""
    br_text = extract_text_features(bug_report.description) + ' ' + extract_text_features(bug_report.summary)
    src_text = extract_text_features(source_file.all_content)
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([br_text, src_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0

def calculate_class_name_similarity(bug_report, source_file):
    """Calculate similarity between class names in bug report and source file"""
    br_text = extract_text_features(bug_report.pos_tagged_description) + ' ' + extract_text_features(bug_report.pos_tagged_summary)
    class_names = ' '.join(source_file.class_names['stemmed'] if isinstance(source_file.class_names, dict) else source_file.class_names)
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([br_text, class_names])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0

def find_previous_reports(file_path, current_time, bug_reports):
    """Find previous bug reports that fixed the same file"""
    previous = []
    for br_id, br in bug_reports.items():
        if br.report_time < current_time and file_path in br.fixed_files:
            previous.append(br)
    return previous

def calculate_collaborative_filtering_score(bug_report, previous_reports):
    """Calculate collaborative filtering score based on previous reports"""
    if not previous_reports:
        return 0.0
    
    similarities = []
    br_text = extract_text_features(bug_report.description) + ' ' + extract_text_features(bug_report.summary)
    
    for prev_report in previous_reports:
        prev_text = extract_text_features(prev_report.description) + ' ' + extract_text_features(prev_report.summary)
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([br_text, prev_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarities.append(similarity)
        except:
            similarities.append(0.0)
    
    return max(similarities) if similarities else 0.0

def calculate_bug_fixing_recency(current_time, previous_reports):
    """Calculate bug fixing recency score"""
    if not previous_reports:
        return 0.0
    
    most_recent = max(br.report_time for br in previous_reports)
    time_diff = (current_time - most_recent).total_seconds()
    return 1.0 / (1.0 + time_diff / (24 * 60 * 60))  # Normalize by days

def extract_features(bug_reports, source_files):
    """Extract features for bug localization"""
    features = []
    total_reports = len(bug_reports)
    
    for i, (br_id, bug_report) in enumerate(bug_reports.items(), 1):
        print(f"Processing bug report {i}/{total_reports}", end='\r')
        
        # For each actual buggy file
        for fixed_file in bug_report.fixed_files:
            if fixed_file not in source_files:
                continue
                
            source_file = source_files[fixed_file]
            
            # Calculate features
            rvsm = calculate_rvsm_similarity(bug_report, source_file)
            class_sim = calculate_class_name_similarity(bug_report, source_file)
            prev_reports = find_previous_reports(fixed_file, bug_report.report_time, bug_reports)
            collab_score = calculate_collaborative_filtering_score(bug_report, prev_reports)
            recency = calculate_bug_fixing_recency(bug_report.report_time, prev_reports)
            frequency = len(prev_reports)
            
            # Add positive sample
            features.append({
                'report_id': br_id,
                'file': fixed_file,
                'rVSM_similarity': rvsm,
                'collab_filter': collab_score,
                'classname_similarity': class_sim,
                'bug_recency': recency,
                'bug_frequency': frequency,
                'match': 1
            })
            
            # Add negative samples (randomly selected non-buggy files)
            non_buggy_files = list(set(source_files.keys()) - set(bug_report.fixed_files))
            if len(non_buggy_files) > 50:
                non_buggy_files = np.random.choice(non_buggy_files, 50, replace=False)
            
            for non_buggy_file in non_buggy_files:
                source_file = source_files[non_buggy_file]
                rvsm = calculate_rvsm_similarity(bug_report, source_file)
                class_sim = calculate_class_name_similarity(bug_report, source_file)
                
                features.append({
                    'report_id': br_id,
                    'file': non_buggy_file,
                    'rVSM_similarity': rvsm,
                    'collab_filter': collab_score,
                    'classname_similarity': class_sim,
                    'bug_recency': recency,
                    'bug_frequency': frequency,
                    'match': 0
                })
    
    return features

def main():
    # Parse and preprocess data
    parser = Parser(DATASET)
    
    print("Parsing and preprocessing bug reports...")
    bug_reports = parser.report_parser()
    report_preprocessor = ReportPreprocessing(bug_reports)
    report_preprocessor.preprocess()
    
    print("Parsing and preprocessing source files...")
    source_files = parser.src_parser()
    src_preprocessor = SrcPreprocessing(source_files)
    src_preprocessor.preprocess()
    
    print("Extracting features...")
    features = extract_features(bug_reports, source_files)
    
    # Save features to CSV
    features_df = pd.DataFrame(features)
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'features_{DATASET.name}.csv')
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

if __name__ == '__main__':
    main() 