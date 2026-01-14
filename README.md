# Comparison of Supervised Learning and Graph-Based Methods for Fake Review Detection

## Overview

This repository contains the implementation and experimental pipeline for an undergraduate thesis entitled:

**"Comparison of Supervised Learning Performance and Graph-Based Methods in Detecting Fake Reviews on E-Commerce"**

The study conducts a controlled and fair comparison between a traditional supervised learning model, **Support Vector Machine (SVM)**, and a graph-based deep learning model, **Graph Convolutional Network (GCN)**, for the task of fake review detection.

The main contribution of this project lies in:

* Using identical text representations (TF-IDF) for both models
* Applying consistent experimental settings
* Performing **single-run and multi-run experiments** with statistical validation
* Analyzing not only predictive performance but also **computational efficiency**

---

## Dataset

* **Source**: Amazon Fake Reviews Dataset (curated by Salminen et al., 2022)
* **Access**: [https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset/data](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset/data)

### Dataset Characteristics

* Total samples: **40,000 reviews**

  * 20,000 genuine reviews (human-written)
  * 20,000 fake reviews (GPT-2 generated)
* Balanced class distribution
* Reviews span **10 Amazon product categories**
* Each review consists of:

  * Product category
  * Rating
  * Review text
  * Label (original / fake)

The dataset is text-centric and does **not** explicitly contain user–product relational data, making it suitable for analyzing the impact of model architecture rather than complex graph semantics.

---

## Project Structure

```bash
.
├── dataset/
│   ├── fake_reviews_dataset.csv      # Raw dataset
│   └── fake_reviews_cleaned.csv      # Preprocessed dataset
├── utility/
│   ├── gcn_util.py                   # GCN model utilities
│   └── svm_util.py                   # SVM model utilities
├── 1_Prepocessing.ipynb              # Text preprocessing & EDA
├── 2A_SVM_Model.ipynb                # SVM training & tuning
├── 2B_GCN_Model.ipynb                # GCN training & tuning
├── 3_Experiment.ipynb                # Comparative experiments & statistics
└── README.md                         # Project documentation
```

---

## Methodology

### 1. Text Preprocessing

* Duplicate removal
* Lowercasing
* Contraction normalization
* Non-ASCII character removal
* Tokenization
* Stopword removal (negations preserved)
* Lemmatization

### 2. Feature Representation

* **TF-IDF** with:

  * `max_features = 5000`
  * Fitted only on training data to avoid data leakage

TF-IDF is intentionally used as a **simple and stable baseline** to ensure that performance differences are driven by model architecture rather than representation complexity.

---

## Experimental Design

Two evaluation scenarios are applied to both models:

1. **Single-Run Evaluation**

   * One execution using fixed hyperparameters
   * Evaluated using Accuracy, F1-score, AUC-ROC, Confusion Matrix, and training time

2. **Multi-Run Evaluation**

   * Five independent runs with different random seeds
   * Mean and standard deviation reported
   * **Paired t-test** applied to assess statistical significance

All experiments use identical data splits to ensure fairness.

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* AUC-ROC
* Training and inference time

---

## Key Findings

* **SVM slightly outperforms GCN** in terms of accuracy and F1-score
* **GCN is significantly more efficient** in computation time
* Both models achieve high AUC (> 0.90), indicating strong discriminative capability
* On text-only datasets with limited relational structure, **traditional machine learning remains highly competitive**

---

## Notes

* This repository is intended for **academic and research purposes**
* Results reflect the characteristics and limitations of the chosen dataset
* The study does not include transformer-based or hybrid models

---

## Author

* **Aqiela Putriana Shabira**
* Undergraduate Program in Data Science
* Faculty of Informatics, Telkom University
* Bandung, Indonesia
