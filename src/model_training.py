import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, precision_score, 
                             recall_score, f1_score, roc_curve, auc)
from imblearn.over_sampling import ADASYN
from sqlalchemy import create_engine
import joblib

def load_data_from_db(db_path="processed_data.db"):
    """Load processed data from SQLite database"""
    engine = create_engine(f"sqlite:///{db_path}")
    read_data = pd.read_sql("SELECT * FROM pca_data", con=engine)
    read_data = read_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return read_data.drop('Target', axis=1).values, read_data['Target'].values

def train_model(X, y):
    """Train and evaluate LightGBM model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Oversampling
    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X_train, y_train)
    
    # Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, -1],
        'min_child_samples': [2, 5],
        'num_leaves': [31, 63],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    lgbm = LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(lgbm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_res, y_res)
    
    return grid_search.best_estimator_, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix.png")
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("results/roc_curve.png")
    plt.close()

if __name__ == "__main__":
    X, y = load_data_from_db()
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_lgbm_model.pkl")
