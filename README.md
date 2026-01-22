# 📊 Customer Churn Prediction Model

## 📌 Overview
Customer churn is a critical challenge for telecom companies. This project builds an **end-to-end machine learning model** to predict whether a customer is likely to churn based on demographics, service usage, contract details, and billing information.

The project follows a **real-world ML workflow**, from data preprocessing to model training, evaluation, and saving the final model for deployment.

---

## 🎯 Problem Statement
To predict customer churn (`Yes/No`) using historical telecom customer data, enabling businesses to take proactive retention actions.

---

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Imbalanced-learn (SMOTE)  
- Joblib  

---

## 📂 Dataset
- **Source:** IBM Telco Customer Churn Dataset  
- **Description:** Telecom customer data including demographics, subscribed services, contract details, charges, and churn status.

### 🎯 Target Variable
- `churn_value`  
  - `1` → Customer churned  
  - `0` → Customer retained  

---

## 🔄 Project Workflow
1. Data loading and cleaning  
2. Dropping irrelevant and leakage columns  
3. Handling missing values  
4. Encoding categorical variables  
5. Train–test split with stratification  
6. Handling class imbalance using **SMOTE**  
7. Feature scaling  
8. Model training  
   - Logistic Regression  
   - Random Forest Classifier  
9. Model evaluation  
   - Confusion Matrix  
   - ROC–AUC Curve  
   - Precision–Recall Curve  
10. Saving the best model as a `.pkl` file  

---

## 📈 Model Evaluation
The models were evaluated using metrics suitable for **imbalanced classification problems**:
- Accuracy  
- F1 Score  
- ROC–AUC  
- Confusion Matrix  

📌 **Random Forest** performed better overall and was selected as the final model.

---

## 💾 Model Saving
The trained model and scaler were saved using `joblib`:

```python
joblib.dump(model, "customer_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
