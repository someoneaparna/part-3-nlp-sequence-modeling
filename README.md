# Part 3: NLP and Sequence Modeling

## Overview
NLP pipeline to classify customer support messages as urgent or not urgent.

## Dataset
Source: [Masai Module 5 Dataset](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
File: `customer_support_text_classification.csv` — 1500 rows, target: `urgent_flag`

## Results
- Baseline (TF-IDF + Logistic Regression): 85% accuracy, Macro F1: 0.83
- LSTM Sequence Model: 85.33% accuracy

## Approach
1. Text cleaned: lowercased, special chars removed, stopwords filtered
2. TF-IDF vectorization (5000 features)
3. Logistic Regression baseline
4. LSTM with Embedding layer for sequence modeling

## Requirements
See requirements.txt
