# =============================================================================
# ĐỒ ÁN: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, 
#        sinh con của giới trẻ Việt Nam (18–35)
# =============================================================================
# File: decision_tree_model.py
# Mô tả: Mô hình Cây quyết định (Decision Tree) với Entropy và Gini
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, RocCurveDisplay,
    classification_report
)
from sklearn.model_selection import cross_val_score
import joblib
import os

from preprocessing import get_tree_preprocessor, FEATURES_REAL, TARGET_REAL


class DecisionTreeModel:
    """
    Class quản lý mô hình Decision Tree cho dữ liệu thực
    """
    
    def __init__(self, criterion="entropy", max_depth=5, min_samples_split=10, 
                 class_weight="balanced", random_state=2026, features=None):
        """
        Khởi tạo mô hình Decision Tree
        
        Parameters:
        -----------
        criterion : str
            Tiêu chí chia ('entropy' hoặc 'gini')
        max_depth : int
            Độ sâu tối đa của cây
        min_samples_split : int
            Số mẫu tối thiểu để chia node
        class_weight : str or dict
            Trọng số các lớp
        random_state : int
            Seed để tái tạo kết quả
        features : list
            Danh sách features (mặc định FEATURES_REAL)
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        self.features = features if features else FEATURES_REAL
        
        # Tạo pipeline
        preprocessor = get_tree_preprocessor(self.features)
        classifier = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state
        )
        
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
        return self
    
    def predict(self, X):
        """
        Dự đoán nhãn
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất
        """
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Đánh giá mô hình
        
        Returns:
        --------
        dict
            Dictionary chứa các chỉ số đánh giá
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5  # Nếu chỉ có 1 class trong test set
            
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """
        Đánh giá mô hình bằng cross-validation
        
        Returns:
        --------
        dict: Kết quả cross-validation
        """
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
    
    def get_feature_names(self):
        """
        Lấy tên các features sau khi encode
        """
        ohe = self.pipeline.named_steps["prep"].named_transformers_["ohe"]
        return ohe.get_feature_names_out(self.features).tolist()
    
    def get_feature_importance(self):
        """
        Lấy feature importance
        
        Returns:
        --------
        pd.DataFrame: Feature importance sorted
        """
        feature_names = self.get_feature_names()
        importances = self.pipeline.named_steps["model"].feature_importances_
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def export_rules(self):
        """
        Xuất luật IF-THEN từ cây
        
        Returns:
        --------
        str
            Chuỗi chứa các luật
        """
        feature_names = self.get_feature_names()
        tree = self.pipeline.named_steps["model"]
        rules = export_text(tree, feature_names=feature_names)
        return rules
    
    def plot_tree(self, figsize=(20, 10), filled=True, class_names=None):
        """
        Vẽ cây quyết định
        """
        if class_names is None:
            class_names = ["Giảm", "Tăng/Giữ nguyên"]
            
        feature_names = self.get_feature_names()
        tree = self.pipeline.named_steps["model"]
        
        plt.figure(figsize=figsize)
        plot_tree(
            tree, 
            feature_names=feature_names,
            class_names=class_names,
            filled=filled,
            rounded=True,
            fontsize=8
        )
        plt.title(f"Decision Tree ({self.criterion.upper()})")
        plt.tight_layout()
        return plt.gcf()
    
    def plot_feature_importance(self, top_n=15, figsize=(10, 6)):
        """
        Vẽ biểu đồ feature importance
        """
        df_importance = self.get_feature_importance().head(top_n)
        
        plt.figure(figsize=figsize)
        bars = plt.barh(df_importance['feature'], df_importance['importance'], color='steelblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - Decision Tree ({self.criterion.upper()})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        return plt.gcf()
    
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
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f"Confusion Matrix - Decision Tree ({self.criterion.upper()})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        return plt.gcf()
    
    def save_model(self, filepath):
        """
        Lưu mô hình
        """
        joblib.dump(self.pipeline, filepath)
        print(f"✓ Đã lưu mô hình vào: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Tải mô hình đã lưu
        """
        pipeline = joblib.load(filepath)
        instance = cls.__new__(cls)
        instance.pipeline = pipeline
        instance.is_fitted = True
        return instance


def train_and_compare_trees(X_train, X_test, y_train, y_test, features=None):
    """
    Huấn luyện và so sánh Decision Tree với entropy và gini
    
    Returns:
    --------
    dict
        Dictionary chứa kết quả của cả 2 mô hình
    """
    results = {}
    
    for criterion in ["entropy", "gini"]:
        print(f"\n{'='*60}")
        print(f"Training Decision Tree with {criterion.upper()}")
        print("="*60)
        
        model = DecisionTreeModel(criterion=criterion, features=features)
        model.fit(X_train, y_train)
        
        eval_results = model.evaluate(X_test, y_test)
        
        print(f"\nKết quả đánh giá:")
        print(f"  Accuracy:  {eval_results['accuracy']:.4f}")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall:    {eval_results['recall']:.4f}")
        print(f"  F1-Score:  {eval_results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {eval_results['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(eval_results['confusion_matrix'])
        
        # Feature Importance
        print(f"\nTop 10 Feature Importance:")
        print(model.get_feature_importance().head(10).to_string(index=False))
        
        results[criterion] = {
            "model": model,
            "metrics": eval_results
        }
    
    return results


def plot_roc_comparison(results, X_test, y_test):
    """
    Vẽ ROC curves so sánh các mô hình
    """
    plt.figure(figsize=(8, 6))
    
    for criterion, data in results.items():
        model = data["model"]
        RocCurveDisplay.from_estimator(
            model.pipeline, 
            X_test, 
            y_test, 
            name=f"Tree-{criterion}"
        )
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title("ROC Curves - Decision Tree Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    from data_loader import create_combined_dataset
    from preprocessing import split_data, FEATURES_REAL
    
    # Load dữ liệu thực
    print("Loading real data...")
    df = create_combined_dataset()
    print(f"Dataset shape: {df.shape}")
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = split_data(df, features=FEATURES_REAL, target=TARGET_REAL)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Huấn luyện và so sánh
    results = train_and_compare_trees(X_train, X_test, y_train, y_test, features=FEATURES_REAL)
    
    # In luật IF-THEN cho mô hình entropy
    print("\n" + "="*60)
    print("LUẬT IF-THEN (Decision Tree - Entropy)")
    print("="*60)
    print(results["entropy"]["model"].export_rules()[:2000])
    
    # Lưu mô hình
    os.makedirs("models", exist_ok=True)
    results["entropy"]["model"].save_model("models/decision_tree_entropy.pkl")
    results["gini"]["model"].save_model("models/decision_tree_gini.pkl")
