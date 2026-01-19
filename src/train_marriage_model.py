# =============================================================================
# HUẤN LUYỆN MÔ HÌNH DỰ BÁO KẾT HÔN - IPUMS DATA
# =============================================================================
# File: train_marriage_model.py
# Mô tả: Huấn luyện các mô hình ML dự báo TÌNH TRẠNG HÔN NHÂN
#        sử dụng dữ liệu IPUMS International Vietnam Census
#
# BIẾN MỤC TIÊU (Target Variable):
#   - Y_married (từ MARST): Tình trạng hôn nhân
#     + 0 = Chưa kết hôn (Single/never married)
#     + 1 = Đã kết hôn (Married/in union)
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CẤU HÌNH
# =============================================================================
DATA_PATH = "data/ipums_processed.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

# Các biến features cho model
CATEGORICAL_FEATURES = [
    'age_group',
    'sex',
    'education_level',
    'urban_rural',
    'region',
    'living_area_level',
    'household_size_group'
]

NUMERICAL_FEATURES = [
    'age',
    'home_ownership',
    'living_area',
    'household_size'
]

TARGET = 'Y_married'

# =============================================================================
# LOAD VÀ CHUẨN BỊ DỮ LIỆU
# =============================================================================
def load_data(file_path: str, sample_size: int = None) -> pd.DataFrame:
    """Load dữ liệu đã xử lý"""
    print(f"Loading data from {file_path}...")
    
    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)
    
    print(f"  Loaded {len(df):,} records")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Chuẩn bị features và target cho model
    
    Returns:
    --------
    X : Features DataFrame
    y : Target Series
    feature_names : List of feature names
    """
    print("Preparing features...")
    
    # Chọn features có trong data
    cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    num_features = [f for f in NUMERICAL_FEATURES if f in df.columns]
    
    print(f"  Categorical features: {cat_features}")
    print(f"  Numerical features: {num_features}")
    
    # One-hot encoding cho categorical
    X = df[cat_features + num_features].copy()
    
    # Encode categorical variables
    for col in cat_features:
        X[col] = X[col].astype(str)
    
    # Get dummies for categorical
    X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=False)
    
    # Target
    y = df[TARGET]
    
    print(f"  Total features after encoding: {X_encoded.shape[1]}")
    
    return X_encoded, y, X_encoded.columns.tolist()


def split_data_by_year(df: pd.DataFrame, test_year: int = 2019):
    """
    Chia dữ liệu theo năm để đánh giá temporal validation
    
    - Train: Năm trước (2009)
    - Test: Năm sau (2019)
    """
    train_df = df[df['year'] != test_year]
    test_df = df[df['year'] == test_year]
    
    print(f"\nTemporal split:")
    print(f"  Train ({train_df['year'].unique()}): {len(train_df):,} records")
    print(f"  Test ({test_df['year'].unique()}): {len(test_df):,} records")
    
    return train_df, test_df


# =============================================================================
# HUẤN LUYỆN MÔ HÌNH
# =============================================================================
def train_decision_tree(X_train, y_train, criterion='entropy', max_depth=10):
    """Huấn luyện Decision Tree"""
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    """Huấn luyện Naive Bayes"""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """Huấn luyện Random Forest"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    """Huấn luyện Logistic Regression"""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# ĐÁNH GIÁ MÔ HÌNH
# =============================================================================
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Đánh giá model và trả về metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
    return metrics


def get_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """Lấy feature importance từ model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return df


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    # Tạo thư mục output nếu chưa có
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dữ liệu
    # Sử dụng sample để chạy nhanh hơn (có thể bỏ sample_size để dùng toàn bộ)
    df = load_data(DATA_PATH, sample_size=500000)
    
    # Chuẩn bị features
    X, y, feature_names = prepare_features(df)
    
    # Chia train/test theo random split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train marriage rate: {y_train.mean():.2%}")
    print(f"Test marriage rate: {y_test.mean():.2%}")
    
    # =========================================================================
    # HUẤN LUYỆN CÁC MÔ HÌNH
    # =========================================================================
    results = []
    models = {}
    
    # 1. Decision Tree (Entropy)
    print("\n" + "="*60)
    print("Training Decision Tree (Entropy)...")
    print("="*60)
    dt_entropy = train_decision_tree(X_train, y_train, criterion='entropy')
    metrics = evaluate_model(dt_entropy, X_test, y_test, "Decision Tree (Entropy)")
    results.append(metrics)
    models['decision_tree_entropy'] = dt_entropy
    
    # 2. Decision Tree (Gini)
    print("\n" + "="*60)
    print("Training Decision Tree (Gini)...")
    print("="*60)
    dt_gini = train_decision_tree(X_train, y_train, criterion='gini')
    metrics = evaluate_model(dt_gini, X_test, y_test, "Decision Tree (Gini)")
    results.append(metrics)
    models['decision_tree_gini'] = dt_gini
    
    # 3. Naive Bayes
    print("\n" + "="*60)
    print("Training Naive Bayes...")
    print("="*60)
    nb = train_naive_bayes(X_train, y_train)
    metrics = evaluate_model(nb, X_test, y_test, "Naive Bayes")
    results.append(metrics)
    models['naive_bayes'] = nb
    
    # 4. Random Forest
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    rf = train_random_forest(X_train, y_train)
    metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
    results.append(metrics)
    models['random_forest'] = rf
    
    # 5. Logistic Regression
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    print("="*60)
    lr = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    results.append(metrics)
    models['logistic_regression'] = lr
    
    # =========================================================================
    # LƯU KẾT QUẢ
    # =========================================================================
    
    # Lưu comparison table
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_ipums.csv", index=False)
    print(f"\nSaved model comparison to {OUTPUT_DIR}/model_comparison_ipums.csv")
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Lưu feature importance
    for name, model in models.items():
        importance_df = get_feature_importance(model, feature_names, name)
        if importance_df is not None:
            importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance_{name}.csv", index=False)
            print(f"\nTop 10 features for {name}:")
            print(importance_df.head(10).to_string(index=False))
    
    # Lưu models
    for name, model in models.items():
        model_path = f"{MODEL_DIR}/{name}_ipums.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")
    
    # Lưu feature names
    joblib.dump(feature_names, f"{MODEL_DIR}/feature_names_ipums.pkl")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    
    return models, results_df


if __name__ == "__main__":
    models, results = main()
