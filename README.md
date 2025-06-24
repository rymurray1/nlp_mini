# Disaster Tweet Classification - NLP Mini Project

## Overview

This project implements a complete Natural Language Processing (NLP) pipeline for classifying tweets as either disaster-related or not. The project is based on the Kaggle competition ["Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started) and provides an end-to-end solution from data preprocessing to model evaluation.

## Problem Statement

The goal is to build a machine learning model that can accurately classify tweets as:
- **Class 0**: Not a disaster tweet
- **Class 1**: Disaster tweet

This is a binary classification problem with real-world applications in emergency response and social media monitoring.

## Dataset

- **Training Data**: 7,613 tweets with labels
- **Test Data**: 3,263 tweets (unlabeled)
- **Features**: 
  - `id`: Unique identifier
  - `keyword`: Relevant keyword (optional)
  - `location`: Tweet location (optional)
  - `text`: Tweet content
  - `target`: Binary label (0 or 1)

## Project Structure

```
week4/
├── nlp_mini_project.ipynb    # Main Jupyter notebook with complete pipeline
├── README.md                 # This file
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
└── submission.csv            # Generated predictions for Kaggle submission
```

## Methodology

### 1. Data Preprocessing
- **Text Cleaning**: Remove URLs, mentions, hashtags, and punctuation
- **Tokenization**: Convert text to tokens using NLTK
- **Lemmatization**: Reduce words to their base form
- **Stop Word Removal**: Remove common English stop words

### 2. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Text length distribution
- Word clouds for disaster vs non-disaster tweets
- Bigram frequency analysis

### 3. Feature Engineering
- **TF-IDF Vectorization**: For classical ML approach
- **Text Tokenization**: For deep learning approach
- **Sequence Padding**: Standardize sequence lengths

### 4. Model Implementation

#### Classical ML Baseline
- **TF-IDF + Logistic Regression**
- Features: 20,000 max features with bigrams
- Class balancing: `class_weight='balanced'`
- Validation Accuracy: ~81%

#### Deep Learning Model
- **Bidirectional LSTM**
- Architecture:
  - Embedding layer (64 dimensions)
  - Bidirectional LSTM (64 units)
  - Global Max Pooling
  - Dense layer (64 units, ReLU)
  - Dropout (0.5)
  - Output layer (sigmoid)
- Validation Accuracy: ~79%

## Key Features

### Text Preprocessing Pipeline
```python
def clean_text(text):
    # Remove URLs, mentions, hashtags
    # Remove punctuation
    # Convert to lowercase
    # Lemmatize and remove stop words
    return cleaned_text
```

### Model Comparison
| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| TF-IDF + Logistic Regression | 0.81 | 0.81 | 0.81 | 81% |
| Bi-LSTM | 0.79 | 0.79 | 0.79 | 79% |

## Setup and Installation

### Prerequisites
- Python 3.10+
- Jupyter Notebook
- Google Colab (recommended for GPU support)

### Dependencies
```bash
pip install kaggle keras-tuner wordcloud nltk emoji
python -m nltk.downloader punkt stopwords wordnet omw-1.4
```

### Required Libraries
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, wordcloud
- **NLP**: nltk
- **Machine Learning**: scikit-learn
- **Deep Learning**: tensorflow, keras
- **Hyperparameter Tuning**: keras-tuner

## Usage

1. **Open the notebook** in Google Colab or Jupyter
2. **Upload Kaggle API credentials** (kaggle.json) when prompted
3. **Run all cells** sequentially
4. **Download submission.csv** for Kaggle submission

## Results and Analysis

### Model Performance
- Both models achieve similar performance (~79-81% accuracy)
- Logistic Regression slightly outperforms Bi-LSTM on this dataset
- Models show good balance between precision and recall

### Error Analysis
- **False Positives**: Political tweets, metaphorical language
- **False Negatives**: Subtle disaster references, movie references

### Key Insights
- Text preprocessing significantly improves model performance
- TF-IDF captures important n-gram patterns effectively
- Bi-LSTM shows potential but may need more data or hyperparameter tuning

## Future Improvements

1. **Advanced Preprocessing**
   - Handle emojis and special characters
   - Implement spell correction
   - Add sentiment analysis features

2. **Model Enhancements**
   - Try transformer models (BERT, RoBERTa)
   - Implement ensemble methods
   - Add cross-validation

3. **Feature Engineering**
   - Include metadata (location, keyword)
   - Add external knowledge bases
   - Implement topic modeling

## Kaggle Competition

This project is designed for the [Kaggle NLP Getting Started competition](https://www.kaggle.com/c/nlp-getting-started). The notebook generates a `submission.csv` file that can be directly uploaded to Kaggle.

## License

This project is for educational purposes as part of the University of Colorado MSCS Deep Learning course.

## Author

Created as part of Week 4 Deep Learning coursework at the University of Colorado.

---

*Note: This project demonstrates fundamental NLP techniques and provides a solid foundation for more advanced text classification tasks.* 