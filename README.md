# 💳 Credit Card Fraud Detection

## 📌 Project Overview
This project applies advanced Machine Learning and Deep Learning techniques to detect fraudulent credit card transactions. The dataset is highly imbalanced, with frauds accounting for only **0.58%** of all transactions.

The goal is to build a classifier that maximizes **Recall** (catching fraud) while maintaining reasonable **Precision** (avoiding false alarms). The project compares traditional ML models (Logistic Regression, Random Forest, XGBoost) against Deep Learning approaches (Neural Networks, TabNet), with a specific focus on handling imbalance using **SMOTE**.

## 📂 Dataset
* **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
* **Structure:** 1,296,675 transactions

## ⚙️ Methodology

### 1. Preprocessing
* **Scaling:** Applied `StandardScaler` to `Amount` and `Time`.
* **Imbalance Handling:** Utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class (Fraud).
* **Deep Learning Specifics:** Implemented **TabNet** (Google's attention-based network for tabular data) to capture complex non-linear relationships.

## 📊 Results Summary

The models were evaluated on **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

| Model | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **TabNet + SMOTE (0.1)** 🏆| **0.54** | **0.64** | **0.59** | **0.98** |
| **TabNet + SMOTE (0.2)** | 0.10 | **0.71** | 0.18 | 0.96 |
| **TabNet (Base)** | **0.83** | 0.31 | 0.46 | 0.95 |
| **Neural Network** | 0.28 | 0.64 | 0.39 | 0.96 |
| **Logistic Regression + SMOTE** | 0.19 | 0.62 | 0.29 | 0.92 |
| **XGBoost (SMOTE + Tuned)** | 0.05 | 0.33 | 0.09 | 0.87 |
| **Random Forest + SMOTE** | 0.35 | 0.07 | 0.12 | 0.85 |

*(Note: Baseline Logistic Regression and Random Forest models struggled significantly with Recall due to the extreme class imbalance.)*

## 📉 Key Insights & Conclusions

### 1. TabNet Superiority
**TabNet** (specifically with SMOTE 0.1) significantly outperformed traditional tree-based models (XGBoost, Random Forest). It achieved the best balance between precision and recall, yielding an **F1-score of 0.59** and a near-perfect **ROC-AUC of 0.98**.

### 2. The Impact of SMOTE
* **Without SMOTE:** Models like Logistic Regression had **0.00 Recall**, completely missing the fraud cases.
* **With SMOTE:** Recall improved drastically across the board. For example, Logistic Regression improved to **0.62 Recall**.
* **Tuning SMOTE:** Increasing the SMOTE ratio to **0.2** (for TabNet) maximized Recall (**0.71**) but caused Precision to plummet to **0.10**, meaning the model flagged too many legitimate transactions as fraud.

### 3. Recommendation
* **Best Balanced Model:** **TabNet + SMOTE-0.1** is the recommended model for deployment as it catches 64% of fraud cases while maintaining over 50% precision.
* **Max Security Model:** If the cost of missing a fraud is extreme, **TabNet + SMOTE-0.2** offers the highest sensitivity (71%), albeit with a higher false positive rate.

## 🛠️ Tech Stack
* **Python** (Pandas, NumPy)
* **Deep Learning:** PyTorch, TabNet (`pytorch-tabnet`)
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
* **Visualization:** Matplotlib, Seaborn
