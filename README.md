# Bug Localization Using DNN and rVSM

This project implements a bug localization system that uses both traditional Information Retrieval (rVSM - revised Vector Space Model) and Deep Neural Networks (DNN) to locate source files that need to be fixed given a bug report. Additionally, it includes a Sentence-BERT (SBERT) based semantic similarity approach.

## Project Structure

```
.
├── data/
│   ├── source-files/     # Contains Java source code files
│   └── bug-reports/      # Contains bug report files in XML/TXT format
├── output/               # Generated features and results
│   ├── features_*.csv    # Extracted features for each dataset
│   ├── results_*.csv     # Evaluation results
│   ├── sbert_similarities_*.csv  # SBERT similarity scores
│   └── sbert_accuracy_*.csv      # SBERT accuracy metrics
├── checkpoints/          # DNN model checkpoints
│   └── model_fold_*.pt   # Saved model states for each fold
├── datasets.py          # Dataset configurations
├── preprocessing.py     # Text preprocessing for bug reports and source code
├── feature_extraction.py # Feature extraction implementation
├── rvsm_model.py       # rVSM model implementation
├── dnn_model.py        # DNN model implementation
├── sbert_similarity.py  # SBERT semantic similarity implementation
├── main.py             # Main script to run the pipeline
└── requirements.txt    # Python dependencies
```

## Features

- **Preprocessing**:
  - Tokenization
  - Camel case splitting
  - Stop word removal
  - Java keyword removal
  - Stemming
  - POS tagging

- **Feature Extraction**:
  - rVSM similarity
  - Class name similarity
  - Collaborative filtering score
  - Bug fixing recency
  - Bug fixing frequency
  - SBERT semantic similarity

- **Models**:
  - rVSM (revised Vector Space Model)
  - DNN (Deep Neural Network)
  - SBERT (Sentence-BERT)

- **Evaluation Metrics**:
  - Top-k Accuracy (k=1,5,10)
  - Mean Average Precision (MAP)
  - Mean Reciprocal Rank (MRR)

## Requirements

- Python 3.6+
- Required packages (install via `pip install -r requirements.txt`):
  - inflection
  - nltk
  - javalang
  - pygments
  - numpy
  - scikit-learn
  - joblib
  - torch
  - pandas
  - sentence-transformers

## Dataset Structure

The system expects data in the following structure:
- `data/source-files/`: Contains Java source code files (either as directories or zip files)
- `data/bug-reports/`: Contains bug report files in XML/TXT format

Supported datasets:
- AspectJ
- Eclipse
- SWT
- Tomcat
- Birt

## Usage

1. **Setup**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd bug-localization-and-report

   # Install dependencies
   pip install -r requirements.txt

   # Download NLTK data
   python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
   ```

2. **Configure Dataset**:
   - Open `datasets.py`
   - Set `DATASET` to your desired dataset (e.g., `DATASET = swt`)

3. **Run the Pipeline**:
   ```bash
   python main.py
   ```

   This will:
   - Extract features if not already done
   - Train and evaluate both rVSM and DNN models
   - Save results to `output/results_<dataset>.csv`

## Model Details

### rVSM Model
- Uses TF-IDF vectorization
- Calculates cosine similarity between bug reports and source files
- Simple but effective baseline model

### DNN Model
- Architecture:
  - Input layer (5 features)
  - Hidden layers (300 -> 200 neurons)
  - Output layer (1 neuron, sigmoid activation)
- Training:
  - Loss: Binary Cross Entropy
  - Optimizer: Adam
  - Early stopping with patience
  - K-fold cross-validation
  - Model checkpointing

### SBERT Model
- Uses pretrained all-MiniLM-L6-v2 model
- Features:
  - Semantic understanding of natural language
  - Contextual embeddings for both bug reports and source code
  - Efficient batch processing
  - GPU acceleration when available
- Process:
  - Combines bug report summary and description
  - Processes source code content, comments, class names, and method names
  - Computes semantic similarity using cosine similarity
  - Ranks files based on similarity scores

## Results

Results are saved in multiple formats:
1. Feature files: `output/features_<dataset>.csv`
   - Contains extracted features for each bug report-source file pair
   - Used for rVSM and DNN models

2. Results file: `output/results_<dataset>.csv`
   - Contains evaluation metrics for rVSM and DNN models
   - Includes Top-k accuracy, MAP, and MRR

3. SBERT results:
   - `output/sbert_similarities_<dataset>.csv`: Detailed similarity scores
   - `output/sbert_accuracy_<dataset>.csv`: Top-k accuracy metrics

## Checkpoints

DNN model checkpoints are saved in the `checkpoints/` directory:
- One checkpoint per fold
- Contains model state, optimizer state, and training loss
- Used for model recovery and analysis

## Contributing

Feel free to contribute to this project by:
- Adding new features
- Implementing new models
- Improving documentation
- Reporting issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on various research papers and existing implementations in the field of bug localization, including:
- "Bug Localization with Combination of Deep Learning and Information Retrieval"
- "Improved Bug Localization using Deep Neural Networks" 