# Authorship Attribution using Language Models

This project implements **authorship attribution** using **n-gram language models** with various smoothing techniques. It trains models on literary texts and classifies unseen text based on **perplexity scores**. The system supports **MLE, Absolute Discounting, Stupid Backoff, Laplace, and Kneser-Ney smoothing**.

## Features
- Train **n-gram language models** for different authors
- Evaluate models using **perplexity-based classification**
- Generate text samples from trained models
- Compare different **smoothing techniques** for better results

## Technologies Used
- **Python** 🐍
- **NLTK** (for language modeling)
- **SpaCy** (for text preprocessing)
- **Scikit-learn** (for train-test splitting)
