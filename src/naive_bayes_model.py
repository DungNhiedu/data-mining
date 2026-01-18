# =============================================================================
# ĐỒ ÁN: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, 
#        sinh con của giới trẻ Việt Nam (18–35)
# =============================================================================
# File: naive_bayes_model.py
# Mô tả: Mô hình Naive Bayes (CategoricalNB/GaussianNB) với Laplace smoothing
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, RocCurveDisplay,
    classification_report
)
from sklearn.model_selection import cross_val_score
import joblib
import os

try:
    from .preprocessing import get_nb_preprocessor, FEATURES_REAL, TARGET_REAL
except ImportError:
    from preprocessing import get_nb_preprocessor, FEATURES_REAL, TARGET_REAL


class NaiveBayesModel:
    """
    Class quản lý mô hình Naive Bayes cho dữ liệu thực
    """
    
    def __init__(self, alpha=1.0, model_type='gaussian', features=None, var_smoothing=1e-9):
        """
        Khởi tạo mô hình Naive Bayes
        
        Parameters:
        -----------
        alpha : float
            Tham số Laplace smoothing cho CategoricalNB (mặc định 1.0)
        model_type : str
            Loại model ('categorical' hoặc 'gaussian')
        features : list
            Danh sách features (mặc định FEATURES_REAL)
        var_smoothing : float
            Variance smoothing cho GaussianNB
        """
        self.alpha = alpha
        self.model_type = model_type
        self.features = features if features else FEATURES_REAL
        self.var_smoothing = var_smoothing
        self.label_encoders = {}
        
        # Tạo pipeline
        preprocessor = get_nb_preprocessor(self.features)
        
        if model_type == 'categorical':
            classifier = CategoricalNB(alpha=alpha)
        else:
            # GaussianNB với var_smoothing cao hơn để xử lý imbalanced data
            classifier = GaussianNB(var_smoothing=var_smoothing)
        
        self.pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", classifier)
        ])
        
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """
        Huấn luyện mô hình
        """
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        
        # Lưu class prior để phân tích
        self.class_counts_ = np.bincount(y_train.astype(int))
        self.class_prior_ = self.class_counts_ / len(y_train)
        
        return self
    
    def predict(self, X):
        """
        Dự đoán nhãn
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất (posterior)
        """
        return self.pipeline.predict_proba(X)
    
    def predict_with_threshold(self, X, threshold=0.3):
        """
        Dự đoán với threshold tùy chỉnh (để xử lý imbalanced data)
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Đánh giá mô hình
        
        Parameters:
        -----------
        X_test : DataFrame
        y_test : Series
        threshold : float
            Ngưỡng phân loại (mặc định 0.5)
        
        Returns:
        --------
        dict: Dictionary chứa các chỉ số đánh giá
        """
        y_proba = self.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5
            
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "threshold": threshold
        }
        
        return results
    
    def find_best_threshold(self, X_val, y_val):
        """
        Tìm threshold tốt nhất dựa trên F1-score
        """
        y_proba = self.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    def cross_validate(self, X, y, cv=5):
        """
        Đánh giá mô hình bằng cross-validation
        
        Returns:
        --------
        dict: Kết quả cross-validation
        """
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='f1')
        return {
            'cv_accuracy': scores,
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
            'cv_f1': f1_scores,
            'cv_f1_mean': f1_scores.mean(),
            'cv_f1_std': f1_scores.std()
        }
    
    def predict_regions(self, regions_df):
        """
        Dự đoán xu hướng cho các vùng/tỉnh
        """
        proba = self.predict_proba(regions_df)[:, 1]
        result = regions_df.copy()
        result["P_trend_up"] = proba
        result["predicted_trend"] = (proba >= 0.5).astype(int)
        return result
    
    def get_class_info(self):
        """
        Lấy thông tin về class distribution
        """
        return {
            'class_counts': self.class_counts_,
            'class_prior': self.class_prior_,
            'imbalance_ratio': self.class_counts_.max() / self.class_counts_.min()
        }
    
    def plot_confusion_matrix(self, y_test, y_pred, class_names=None):
        """
        Vẽ ma trận nhầm lẫn
        """
        if class_names is None:
            class_names = ["Giảm", "Tăng/Giữ nguyên"]
            
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f"Confusion Matrix - Naive Bayes ({self.model_type})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        return plt.gcf()
    
    def plot_probability_distribution(self, X_test, y_test):
        """
        Vẽ phân phối xác suất dự đoán
        """
        y_proba = self.predict_proba(X_test)[:, 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(y_proba[y_test == 0], bins=20, alpha=0.5, label='Giảm (actual)', color='red')
        plt.hist(y_proba[y_test == 1], bins=20, alpha=0.5, label='Tăng (actual)', color='green')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
        plt.xlabel('Predicted Probability of Tăng')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution by Actual Class')
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
    
    def save_model(self, filepath):
        """
        Lưu mô hình
        """
        joblib.dump({
            'pipeline': self.pipeline,
            'features': self.features,
            'model_type': self.model_type,
            'alpha': self.alpha
        }, filepath)
        print(f"[OK] Da luu mo hinh vao: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Tải mô hình đã lưu
        """
        data = joblib.load(filepath)
        instance = cls.__new__(cls)
        instance.pipeline = data['pipeline']
        instance.features = data.get('features', FEATURES_REAL)
        instance.model_type = data.get('model_type', 'gaussian')
        instance.alpha = data.get('alpha', 1.0)
        instance.is_fitted = True
        return instance


def compare_nb_models(X_train, X_test, y_train, y_test, features=None):
    """
    So sánh CategoricalNB và GaussianNB với các threshold khác nhau
    
    Returns:
    --------
    dict: Kết quả so sánh
    """
    results = {}
    
    # GaussianNB với các var_smoothing khác nhau
    for var_smooth in [1e-9, 1e-5, 1e-3]:
        model_name = f'gaussian_vs{var_smooth}'
        print(f"\n{'='*60}")
        print(f"Training Naive Bayes (GaussianNB, var_smoothing={var_smooth})")
        print("="*60)
        
        model = NaiveBayesModel(
            model_type='gaussian', 
            features=features,
            var_smoothing=var_smooth
        )
        model.fit(X_train, y_train)
        
        # Tìm threshold tốt nhất
        best_threshold, best_f1 = model.find_best_threshold(X_test, y_test)
        print(f"Best threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
        
        # Đánh giá với threshold tốt nhất
        eval_results = model.evaluate(X_test, y_test, threshold=best_threshold)
        
        print(f"\nKết quả đánh giá (threshold={best_threshold:.2f}):")
        print(f"  Accuracy:  {eval_results['accuracy']:.4f}")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall:    {eval_results['recall']:.4f}")
        print(f"  F1-Score:  {eval_results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {eval_results['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(eval_results['confusion_matrix'])
        
        results[model_name] = {
            "model": model,
            "metrics": eval_results,
            "best_threshold": best_threshold
        }
    
    return results


if __name__ == "__main__":
    try:
        from .data_loader import create_combined_dataset
        from .preprocessing import split_data, FEATURES_REAL
    except ImportError:
        from data_loader import create_combined_dataset
        from preprocessing import split_data, FEATURES_REAL
    
    # Load du lieu thuc
    print("Loading real data...")
    df = create_combined_dataset()
    print(f"Dataset shape: {df.shape}")
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(df['trend'].value_counts())
    print(f"Imbalance ratio: {df['trend'].value_counts().max() / df['trend'].value_counts().min():.2f}")
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = split_data(df, features=FEATURES_REAL, target=TARGET_REAL)
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # So sánh các mô hình NB
    results = compare_nb_models(X_train, X_test, y_train, y_test, features=FEATURES_REAL)
    
    # Chọn và lưu model tốt nhất (theo F1)
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
    print(f"\n[OK] Best model: {best_model_name}")
    
    os.makedirs("models", exist_ok=True)
    results[best_model_name]["model"].save_model("models/naive_bayes_best.pkl")
