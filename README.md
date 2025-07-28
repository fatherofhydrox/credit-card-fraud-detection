# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning classification techniques. The goal is to build a reliable predictive model that distinguishes between genuine and fraudulent transactions in a highly imbalanced dataset.

## 📌 Problem Statement

Credit card fraud is a significant financial crime that needs accurate and timely detection. The challenge lies in identifying the rare fraud cases from a vast number of legitimate transactions without causing false alarms.

## 📂 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Description: Contains transactions made by European cardholders in September 2013. 
- Records: 284,807
- Fraud Cases: 492 (~0.17% of total)
- Features: `V1` to `V28` are anonymized PCA components; `Time`, `Amount`, and `Class` (0 = legit, 1 = fraud)

## 🛠️ Technologies Used

- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Model:** Logistic Regression, Random Forest, Decision Tree
- **Tools:** Jupyter Notebook

## 🔍 Exploratory Data Analysis (EDA)

- Visualized class imbalance
- Detected outliers
- Analyzed correlation heatmaps
- Scaled features using `StandardScaler`

## ⚙️ Model Building & Evaluation

| Model              | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.91      | 0.63   | 0.74     | 0.95    |
| Random Forest       | 0.99      | 0.76   | 0.86     | 0.97    |
| Decision Tree       | 0.91      | 0.74   | 0.82     | 0.93    |

> ✅ **Random Forest** gave the best overall performance with a good balance between precision and recall.

## 🧠 Techniques Applied

- **Handling Imbalanced Data:**
  - Undersampling and SMOTE (Synthetic Minority Oversampling)
- **Cross-validation:** Stratified K-Fold
- **Evaluation Metrics:** Confusion Matrix, ROC-AUC, Precision-Recall Curve

## 📈 Key Insights

- Highly imbalanced dataset requires careful evaluation metrics.
- Feature scaling improved model accuracy.
- Random Forest is robust for fraud detection due to its ensemble nature.

## 🚀 Future Work

- Integrate with real-time transaction APIs.
- Use Deep Learning (Autoencoders / LSTM).
- Deploy using Flask or Streamlit for live prediction.

## 📎 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/fatherofhydrox/credit-card-fraud-detection
   cd credit-card-fraud-detection
pip install -r requirements.txt

jupyter notebook fraud_detection.ipynb

🙌 Acknowledgements
Dataset by ULB Machine Learning Group
Inspired by the practical need for fraud detection in financial systems

🚀 **Live Demo:** [Credit Card Fraud Detection App](https://credit-card-fraud-detection-1-jbig.onrender.com)
