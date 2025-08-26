# Translational Natural Language Inference (NLI) Project

## Overview

This project explores Natural Language Inference (NLI) in a multilingual setting by evaluating the performance of translated English premise and hypothesis pairs. The project uses state-of-the-art machine translation techniques to convert English pairs into French and then performs NLI classification tasks to predict logical relationships between the statements.

## Project Structure

```
Translational_NLI/
├── Classification_Final/          # NLI classification models and pipeline
├── Translation_Final/             # Machine translation implementation
├── Evaluation_Final/              # Model evaluation and analysis
├── Output/                        # Results and predictions
└── README.md                      # Project documentation
```

## Key Components

### 1. Classification Pipeline (`Classification_Final/`)

The classification system implements multiple approaches for Natural Language Inference:

- **Traditional ML Models**: Multinomial Naive Bayes classifier
- **Deep Learning Models**: 
  - CNN (Convolutional Neural Network) with custom architecture
  - Transformer-based models (BERT, RoBERTa, ELECTRA, XLNet)
- **Data Processing**: Text preprocessing, tokenization, and Word2Vec embeddings
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Cohen's Kappa

#### Files:
- `main.py`: Main execution script for training and evaluation
- `model.py`: CNN model architecture implementation
- `preprocessing.py`: Text preprocessing and data pipeline
- `dataloader.py`: PyTorch data loading utilities
- `fever_nli_french_classification_BR.ipynb`: BERT/RoBERTa classification notebook
- `fever_nli_french_classification_EX.ipynb`: ELECTRA/XLNet classification notebook

### 2. Translation System (`Translation_Final/`)

Implements sequence-to-sequence models for English to French translation:

- **Model Architecture**: Transformer-based seq2seq models
- **Training Data**: FEVER dataset with English premises/hypotheses and French references
- **Implementation**: Separate models for premises and hypotheses translation
- **Dataset Sizes**: 1000 and 2000 sample configurations

#### Files:
- `seq_2_seq_premises_1000.ipynb`: Premise translation with 1000 samples
- `seq_2_seq_premises_2000.ipynb`: Premise translation with 2000 samples  
- `seq_2_seq_hypothesis_2000.ipynb`: Hypothesis translation with 2000 samples

### 3. Evaluation Framework (`Evaluation_Final/`)

Comprehensive evaluation and analysis of model performance:

- **Performance Comparison**: Original vs. predicted French text performance
- **Misclassification Analysis**: Detailed error analysis and patterns
- **Sentiment Analysis**: Semantic similarity and sentiment evaluation
- **Final Reports**: Comprehensive classification performance reports

#### Files:
- `Final Classification Report - MSCI Project.ipynb`: Main evaluation notebook
- `Misclassification_calculation.ipynb`: Error analysis and patterns
- `Sentiment_semantic.ipynb`: Semantic and sentiment evaluation
- `process_csv.ipynb`: Data processing utilities

### 4. Output and Results (`Output/`)

Contains all model outputs and evaluation results:

- **Original Results**: Performance on original French text
- **Predicted Results**: Performance on machine-translated text
- **Comparison Data**: Side-by-side performance metrics
- **Prediction Files**: Model predictions for all test samples

## Dataset

The project uses the **FEVER (Fact Extraction and Verification)** dataset:
- **Original Language**: English premises and hypotheses
- **Target Language**: French translations
- **Task**: Natural Language Inference (Entailment, Contradiction, Neutral)
- **Format**: Parquet files with premise-hypothesis-label triples

## Model Performance

The project evaluates multiple model architectures:

### Traditional ML Models
- **Multinomial Naive Bayes**: Baseline performance on French text
- **CNN**: Custom convolutional architecture with Word2Vec embeddings

### Transformer Models
- **BERT**: Bidirectional Encoder Representations from Transformers
- **RoBERTa**: Robustly Optimized BERT Pretraining Approach
- **ELECTRA**: Efficiently Learning an Encoder that Classifies Token Replacements Accurately
- **XLNet**: Generalized Autoregressive Pretraining for Language Understanding

## Usage

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn gensim nltk pyarrow tensorflow
```

### Running Classification
```bash
cd Classification_Final
python main.py
```

### Running Translation
```bash
cd Translation_Final
# Open and run the appropriate Jupyter notebook
jupyter notebook seq_2_seq_premises_2000.ipynb
```

### Evaluation
```bash
cd Evaluation_Final
# Open and run the evaluation notebooks
jupyter notebook "Final Classification Report - MSCI Project.ipynb"
```

## Key Findings

1. **Translation Quality Impact**: Machine translation quality significantly affects NLI performance
2. **Model Robustness**: Transformer models show better cross-lingual transfer capabilities
3. **Language-Specific Patterns**: French language characteristics influence classification accuracy
4. **Data Augmentation**: Translated data can be used for multilingual NLI training

## Research Contributions

- **Multilingual NLI**: Novel approach to cross-lingual natural language inference
- **Translation-NLI Pipeline**: End-to-end system for multilingual text understanding
- **Performance Analysis**: Comprehensive evaluation of translation impact on NLI tasks
- **Practical Applications**: Real-world applications in cross-lingual information retrieval

## Future Work

- **Advanced Translation Models**: Integration of state-of-the-art translation systems
- **Multilingual Training**: Joint training on multiple languages
- **Domain Adaptation**: Specialized models for specific domains
- **Real-time Processing**: Optimization for real-time multilingual NLI applications