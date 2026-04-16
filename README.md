# Fake News Detector

A machine learning project for detecting fake news in English-language news articles using classical NLP techniques and scikit-learn models.

## Project Description

This project classifies news articles into two categories:

- **FAKE** (`0`)
- **REAL** (`1`)

The workflow includes:

- data loading and preprocessing,
- baseline model comparison,
- hyperparameter tuning,
- final model evaluation,
- model export,
- CLI-based prediction for new text input.

The project uses article titles and article bodies, combines them into a single text feature, transforms the text using **TF-IDF**, and then trains a classifier for fake news detection.

## Features

- preprocessing of raw CSV datasets,
- validation of required columns,
- train/test split with stratification,
- baseline comparison using:
  - Logistic Regression,
  - LinearSVC,
  - Complement Naive Bayes,
- model selection based on **F1 Macro**,
- hyperparameter tuning with `GridSearchCV`,
- final evaluation on a held-out test set,
- classification report and confusion matrix,
- trained model export with `joblib`,
- simple command-line prediction script.

## Project Structure

FakeNewsDetector/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   └── fake_news_model.joblib
│
├── reports/
│   ├── baseline_results.csv
│   ├── baseline_results.json
│   ├── gridsearch_results.csv
│   └── final_metrics.json
│
├── data_utils.py
├── train_baselines.py
├── tune_and_save.py
├── predict.py
├── requirements.txt
└── README.md


## Dataset Format

The project expects two CSV files inside the data/ directory:
    Fake.csv
    True.csv

Each file should contain at least these columns:
    title
    text

The label column is created automatically in code:
    0 = fake
    1 = real


## Workflow

1. Data preprocessing
data_utils.py is responsible for:
    validating required columns,
    filling missing values,
    combining title and text into one content column,
    cleaning text,
    preparing the final DataFrame,
    splitting data into train and test sets.


2. Baseline model comparison
train_baselines.py:
    loads and preprocesses the dataset,
    uses the training split,
    compares baseline models with cross-validation,
    saves the ranking of models to reports/baseline_results.csv,
    saves the selected best baseline to reports/baseline_results.json.


3. Hyperparameter tuning and model saving
tune_and_save.py:
    reads the best baseline model,
    runs GridSearchCV,
    evaluates the best estimator on the test set,
    saves tuning results to reports/gridsearch_results.csv,
    saves final metrics to reports/final_metrics.json,
    saves the trained model to models/fake_news_model.joblib.


4. Prediction
predict.py:
    loads the trained model,
    asks the user for input text,
    preprocesses the text,
    predicts whether the news is FAKE or REAL,
    displays class probabilities when available.


## Installation

Clone the repository and install dependencies:

git clone https://github.com/adrianstreciwilk19-jpg/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt

Usage
Step 1 — Compare baseline models
    python train_baselines.py

Step 2 — Tune and save the best model
    python tune_and_save.py

Step 3 — Run prediction
    python predict.py

Example Prediction
Wpisz tekst do analizy.
Tekst: Breaking news article text goes here...

=== WYNIK ===
Klasyfikacja: REAL
Prawdopodobieństwo FAKE: 0.1234 (12.34%)
Prawdopodobieństwo REAL: 0.8766 (87.66%)

Output Files
    After training, the project generates the following files:

In reports/
    baseline_results.csv — baseline comparison results
    baseline_results.json — selected best baseline
    gridsearch_results.csv — full GridSearchCV results
    final_metrics.json — final evaluation metrics

In models/
    fake_news_model.joblib — trained final model

Technologies Used
    Python
    pandas
    scikit-learn
    joblib

Possible Future Improvements
    support for additional datasets,
    more advanced text normalization,
    threshold tuning for prediction confidence,
    web API with Flask or FastAPI,
    simple frontend or web interface,
    comparison with transformer-based models,
    experiment tracking.