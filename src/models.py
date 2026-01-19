# =============================================================================
# ĐỒ ÁN: Dự báo Xu hướng Kết hôn của Giới trẻ Việt Nam (18-35)
# =============================================================================
# File: models.py
# Mô tả: Định nghĩa và huấn luyện các mô hình Machine Learning
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import joblib
import os


class MarriagePredictionModels:
    """
    Lớp chứa các mô hình dự báo xu hướng kết hôn
    """
    
    def __init__(self):
        self.models = {
            'Decision Tree (Entropy)': DecisionTreeClassifier(
                criterion='entropy',
                max_depth=10,
                min_samples_split=100,
                min_samples_leaf=50,
                random_state=42
            ),
            'Decision Tree (Gini)': DecisionTreeClassifier(
                criterion='gini',
                max_depth=10,
                min_samples_split=100,
                min_samples_leaf=50,
                random_state=42
            ),
            'Naive Bayes': GaussianNB()
        }
        self.trained_models = {}
        self.results = {}
    
    def train_all(self, X_train, y_train):
        """Huấn luyện tất cả mô hình"""
        print("🚀 Bắt đầu huấn luyện các mô hình...")
        
        for name, model in self.models.items():
            print(f"  [TRAINING] Dang huan luyen: {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            print(f"  [OK] Hoan thanh: {name}")
        
        print("🎉 Huấn luyện hoàn tất tất cả mô hình!")
        return self.trained_models
    
    def evaluate_all(self, X_test, y_test):
        """Đánh giá tất cả mô hình"""
        print("\n[EVAL] Danh gia cac mo hinh...")
        
        results = []
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            }
            
            if y_prob is not None:
                metrics['AUC-ROC'] = roc_auc_score(y_test, y_prob)
            else:
                metrics['AUC-ROC'] = None
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['True Negative'] = cm[0, 0]
            metrics['False Positive'] = cm[0, 1]
            metrics['False Negative'] = cm[1, 0]
            metrics['True Positive'] = cm[1, 1]
            
            results.append(metrics)
            
            print(f"  {name}:")
            print(f"    Accuracy: {metrics['Accuracy']:.4f}")
            print(f"    F1-Score: {metrics['F1-Score']:.4f}")
            if metrics['AUC-ROC']:
                print(f"    AUC-ROC: {metrics['AUC-ROC']:.4f}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def cross_validate(self, X, y, cv=5):
        """Cross-validation cho tất cả mô hình"""
        print(f"\n🔄 Cross-Validation ({cv} folds)...")
        
        cv_results = []
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_results.append({
                'Model': name,
                'CV Mean Accuracy': scores.mean(),
                'CV Std': scores.std()
            })
            print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return pd.DataFrame(cv_results)
    
    def get_feature_importance(self, feature_names):
        """Lấy feature importance từ các mô hình hỗ trợ"""
        importance_data = []
        
        for name, model in self.trained_models.items():
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(feature_names, model.feature_importances_):
                    importance_data.append({
                        'Model': name,
                        'Feature': feat,
                        'Importance': imp
                    })
        
        return pd.DataFrame(importance_data)
    
    def get_decision_tree_rules(self, feature_names, model_name='Decision Tree (Entropy)', max_depth=3):
        """Trích xuất luật IF-THEN từ Decision Tree"""
        from sklearn.tree import export_text
        
        if model_name in self.trained_models:
            model = self.trained_models[model_name]
            if isinstance(model, DecisionTreeClassifier):
                rules = export_text(model, feature_names=feature_names, max_depth=max_depth)
                return rules
        return None
    
    def save_models(self, output_dir='models'):
        """Lưu các mô hình đã huấn luyện"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"  💾 Đã lưu: {filepath}")
    
    def load_models(self, input_dir='models'):
        """Tải các mô hình đã lưu"""
        for name in self.models.keys():
            filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
            filepath = os.path.join(input_dir, filename)
            if os.path.exists(filepath):
                self.trained_models[name] = joblib.load(filepath)
                print(f"  📂 Đã tải: {name}")
    
    def predict(self, X, model_name='Decision Tree (Entropy)'):
        """Dự báo với mô hình được chọn"""
        if model_name in self.trained_models:
            model = self.trained_models[model_name]
            prediction = model.predict(X)
            probability = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            return prediction, probability
        return None, None


def get_roc_curves(models, X_test, y_test):
    """Tính ROC curve cho các mô hình"""
    roc_data = {}
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            roc_data[name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc
            }
    
    return roc_data


if __name__ == "__main__":
    # Test
    from data_processor import load_and_process_data, prepare_features, split_data
    
    df = load_and_process_data(sample_size=50000)
    X, y, feature_names, encoders = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    models = MarriagePredictionModels()
    models.train_all(X_train, y_train)
    results = models.evaluate_all(X_test, y_test)
    print("\n[RESULT] Ket qua so sanh:")
    print(results)
