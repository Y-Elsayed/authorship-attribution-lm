# Authorship Attribution using Language Models

This project implements **authorship attribution** using **n-gram language models** with various smoothing techniques. It trains models on literary texts and classifies unseen text based on **perplexity scores**. The system supports **MLE, Absolute Discounting, Stupid Backoff, Laplace, and Kneser-Ney smoothing**.

## Features
- Train **n-gram language models** for different authors
- Evaluate models using **perplexity-based classification**
- Generate text samples from trained models
- Compare different **smoothing techniques** for better results

# Authorship Attribution using N-gram Language Models

## Key Features
- Custom n-gram language model implementation without NLTK
- NLTK-based n-gram models for comparison
- Transformer-based sequence classifier using Hugging Face
- Various smoothing techniques (MLE, Laplace, Stupid Backoff, Kneser-Ney)
- Text generation capabilities
- Feature extraction for author style analysis

## File Structure
- **classifier.py**: Main command-line interface that handles all approaches
- **data_processor.py**: Handles text preprocessing and data splitting
- **ngram_authorship_classifier.py**: Custom n-gram implementation (without NLTK)
- **ngram_authorship_classifier_nltk.py**: N-gram implementation using NLTK
- **ngram_helper.py**: Helper functions for custom n-gram implementation
- **hf_sequence_classifier.py**: Sequence classifier using Hugging Face transformers
- **authors_dataset.py**: Dataset class for the sequence classifier

### Notebooks
- **testing_ngram_models.ipynb**: Testing custom n-gram models
- **testing_ngram_models_nltk.ipynb**: Testing NLTK-based n-gram models
- **training_sequence_classifier.ipynb**: Training the sequence classifier
- **evaluating_sequence_classifier.ipynb**: Evaluating the sequence classifier
- **ngram_generate_samples.ipynb**: Generating text samples from n-gram models

## Usage

#### Requirements
- Python Version 3.10.11
- Install the requirements.txt : `pip install -r requirements.txt`
  
The main script `classifier.py` can be run with the following commands:

```bash
# Train and evaluate on development set
python3 classifier.py authorlist -approach [generative|discriminative]

# Train and predict on test file
python3 classifier.py authorlist -approach [generative|discriminative] -test testfile
```

## Team Members
- MJ Corey (corey094@umn.edu)
- Yassin Ali (ali00740@umn.edu)
- Jordan Johnson (joh20376@umn.edu)
- Akshat Ghoshal (ghosh159@umn.edu)
