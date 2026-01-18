# =============================================================================
# ĐỒ ÁN: Dự báo Quyết định Kết Hôn trong Năm (18–35), Giai đoạn 2019–2024
# =============================================================================
# File: marriage_prediction_model.py
# Mô tả: Mô hình Decision Tree và Naive Bayes cho dự báo kết hôn
# =============================================================================

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, classification_report
)


# Định nghĩa features
FEATURES_TREE_CAT = [
    "age_group", "sex", "region", "urban_rural", "education",
    "employment_status", "home_ownership", "income_level", "afford_level"
]

FEATURES_TREE_NUM = [
    "affordability_index", "unemployment_rate",
    "house_price_index", "family_tax_benefit_index"
]

FEATURES_NB = [
    "age_group", "sex", "region", "urban_rural", "education",
    "employment_status", "home_ownership", "income_level", "afford_level",
    "unemp_level", "hpi_level", "tax_level"
]

TARGET = "Y_married"


def prepare_data(panel_path="data/panel_microdata.csv"):
    """Chuẩn bị dữ liệu cho training"""
    panel = pd.read_csv(panel_path)
    
    # Chuyển đổi categorical
    for col in ["age_group", "income_level", "afford_level"]:
        if col in panel.columns:
            panel[col] = panel[col].astype(str)
    
    # Tạo thêm các biến binned cho Naive Bayes
    panel["unemp_level"] = pd.qcut(
        panel["unemployment_rate"].rank(method="first"), 
        4, 
        labels=["Low", "Med-Low", "Med-High", "High"]
    )
    panel["hpi_level"] = pd.qcut(
        panel["house_price_index"].rank(method="first"), 
        4, 
        labels=["Low", "Med-Low", "Med-High", "High"]
    )
    panel["tax_level"] = pd.qcut(
        panel["family_tax_benefit_index"].rank(method="first"), 
        4, 
        labels=["Low", "Med-Low", "Med-High", "High"]
    )
    
    # Chia theo năm
    train_df = panel[panel.year.isin([2019, 2020, 2021, 2022])]
    val_df = panel[panel.year == 2023]
    test_df = panel[panel.year == 2024]
    
    return panel, train_df, val_df, test_df


def train_decision_tree(train_df, val_df, test_df, criterion="entropy", save_path=None):
    """Huấn luyện Decision Tree"""
    
    # Chuẩn bị features
    features = FEATURES_TREE_CAT + FEATURES_TREE_NUM
    
    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_val = val_df[features]
    y_val = val_df[TARGET]
    X_test = test_df[features]
    y_test = test_df[TARGET]
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), FEATURES_TREE_CAT)
    ], remainder="passthrough")
    
    # Model
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=6,
        min_samples_split=120,
        class_weight="balanced",
        random_state=2026
    )
    
    # Pipeline
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", clf)
    ])
    
    # Fit
    pipe.fit(X_train, y_train)
    
    # Evaluate
    results = {}
    for X, y, tag in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        y_pred = pipe.predict(X)
        y_proba = pipe.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")
        auc = roc_auc_score(y, y_proba)
        
        results[tag] = {
            "accuracy": acc,
            "precision": pr,
            "recall": rc,
            "f1": f1,
            "roc_auc": auc
        }
        
        print(f"[{criterion}][{tag}] Acc={acc:.3f} P={pr:.3f} R={rc:.3f} F1={f1:.3f} AUC={auc:.3f}")
    
    # Export rules
    feat_names = (
        pipe.named_steps["prep"]
        .named_transformers_["ohe"]
        .get_feature_names_out(FEATURES_TREE_CAT).tolist()
        + FEATURES_TREE_NUM
    )
    rules = export_text(pipe.named_steps["model"], feature_names=feat_names, max_depth=4)
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipe, save_path)
        print(f"Saved model to {save_path}")
    
    return pipe, results, rules


def train_naive_bayes(train_df, val_df, test_df, alpha=1.0, save_path=None):
    """Huấn luyện Naive Bayes"""
    from sklearn.naive_bayes import GaussianNB
    
    X_train = train_df[FEATURES_NB]
    y_train = train_df[TARGET]
    X_val = val_df[FEATURES_NB]
    y_val = val_df[TARGET]
    X_test = test_df[FEATURES_NB]
    y_test = test_df[TARGET]
    
    # Preprocessor - sử dụng OrdinalEncoder
    preprocessor = ColumnTransformer([
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999), FEATURES_NB)
    ], remainder="drop")
    
    # Pipeline - Sử dụng GaussianNB thay vì CategoricalNB để tránh lỗi với encoded values
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", GaussianNB(var_smoothing=alpha * 1e-9))
    ])
    
    # Fit
    pipe.fit(X_train, y_train)
    
    # Evaluate
    results = {}
    for X, y, tag in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        y_pred = pipe.predict(X)
        y_proba = pipe.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")
        auc = roc_auc_score(y, y_proba)
        
        results[tag] = {
            "accuracy": acc,
            "precision": pr,
            "recall": rc,
            "f1": f1,
            "roc_auc": auc
        }
        
        print(f"[NaiveBayes][{tag}] Acc={acc:.3f} P={pr:.3f} R={rc:.3f} F1={f1:.3f} AUC={auc:.3f}")
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipe, save_path)
        print(f"Saved model to {save_path}")
    
    return pipe, results


def predict_personas(model, personas_df, model_type="tree"):
    """Dự báo cho các personas"""
    
    if model_type == "tree":
        features = FEATURES_TREE_CAT + FEATURES_TREE_NUM
    else:
        features = FEATURES_NB
    
    X = personas_df[features]
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    
    result = personas_df.copy()
    result["P_married"] = proba
    result["prediction"] = pred
    
    return result


def get_sample_personas():
    """Tạo mẫu personas để dự báo"""
    personas = pd.DataFrame({
        "age_group": ["25-29", "25-29", "30-35", "18-24", "30-35"],
        "sex": ["Nữ", "Nam", "Nữ", "Nữ", "Nam"],
        "region": ["Nam", "Bắc", "Bắc", "Trung", "Nam"],
        "urban_rural": ["Đô thị", "Đô thị", "Nông thôn", "Đô thị", "Nông thôn"],
        "education": ["ĐH", "ĐH", "≤THPT", ">ĐH", "≤THPT"],
        "employment_status": ["FT", "FT", "FT", "PT", "FT"],
        "home_ownership": [0, 0, 1, 0, 1],
        "income_level": ["mid", "mid", "low", "high", "low"],
        "afford_level": ["Q2", "Q2", "Q3", "Q1", "Q4"],
        "affordability_index": [0.8, 0.85, 1.2, 0.5, 1.5],
        "unemployment_rate": [3.5, 2.8, 2.5, 3.2, 3.0],
        "house_price_index": [125, 110, 105, 130, 100],
        "family_tax_benefit_index": [110, 115, 118, 108, 120],
        "unemp_level": ["Med-High", "Med-Low", "Low", "Med-High", "Med-Low"],
        "hpi_level": ["High", "Med-High", "Med-Low", "High", "Low"],
        "tax_level": ["Med-Low", "Med-High", "High", "Low", "High"]
    })
    return personas


def extract_if_then_rules(pipe, max_rules=10):
    """Trích xuất luật IF-THEN từ Decision Tree"""
    
    tree = pipe.named_steps["model"]
    feat_names = (
        pipe.named_steps["prep"]
        .named_transformers_["ohe"]
        .get_feature_names_out(FEATURES_TREE_CAT).tolist()
        + FEATURES_TREE_NUM
    )
    
    rules = []
    
    def recurse(node, conditions=[]):
        if tree.tree_.feature[node] != -2:  # Not a leaf
            feature = feat_names[tree.tree_.feature[node]]
            threshold = tree.tree_.threshold[node]
            
            # Left branch (<=)
            left_conditions = conditions + [f"{feature} <= {threshold:.2f}"]
            recurse(tree.tree_.children_left[node], left_conditions)
            
            # Right branch (>)
            right_conditions = conditions + [f"{feature} > {threshold:.2f}"]
            recurse(tree.tree_.children_right[node], right_conditions)
        else:
            # Leaf node
            values = tree.tree_.value[node][0]
            total = sum(values)
            class_0_prob = values[0] / total
            class_1_prob = values[1] / total
            prediction = "Kết hôn" if class_1_prob > 0.5 else "Không kết hôn"
            confidence = max(class_0_prob, class_1_prob)
            
            if len(conditions) <= 4 and confidence > 0.6:  # Chỉ lấy luật ngắn và tin cậy
                rules.append({
                    "conditions": " AND ".join(conditions),
                    "prediction": prediction,
                    "confidence": confidence,
                    "samples": int(total)
                })
    
    recurse(0)
    
    # Sắp xếp theo confidence và lấy top rules
    rules = sorted(rules, key=lambda x: (-x["confidence"], -x["samples"]))[:max_rules]
    
    return rules


if __name__ == "__main__":
    # Prepare data
    print("Loading data...")
    panel, train_df, val_df, test_df = prepare_data()
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Y_married rate - Train: {train_df[TARGET].mean():.3f}, Val: {val_df[TARGET].mean():.3f}, Test: {test_df[TARGET].mean():.3f}")
    
    # Train Decision Tree (Entropy)
    print("\n" + "="*50)
    print("Training Decision Tree (Entropy)...")
    tree_entropy, results_entropy, rules_entropy = train_decision_tree(
        train_df, val_df, test_df, 
        criterion="entropy",
        save_path="models/tree_entropy_panel.pkl"
    )
    
    # Train Decision Tree (Gini)
    print("\n" + "="*50)
    print("Training Decision Tree (Gini)...")
    tree_gini, results_gini, rules_gini = train_decision_tree(
        train_df, val_df, test_df,
        criterion="gini", 
        save_path="models/tree_gini_panel.pkl"
    )
    
    # Train Naive Bayes
    print("\n" + "="*50)
    print("Training Naive Bayes...")
    nb_model, results_nb = train_naive_bayes(
        train_df, val_df, test_df,
        alpha=1.0,
        save_path="models/naive_bayes_panel.pkl"
    )
    
    # Predict personas
    print("\n" + "="*50)
    print("Predicting for personas...")
    personas = get_sample_personas()
    
    result_tree = predict_personas(tree_entropy, personas, "tree")
    result_nb = predict_personas(nb_model, personas, "nb")
    
    print("\nTree predictions:")
    print(result_tree[["age_group", "urban_rural", "education", "afford_level", "P_married"]])
    
    print("\nNaive Bayes predictions:")
    print(result_nb[["age_group", "urban_rural", "education", "afford_level", "P_married"]])
    
    # Extract IF-THEN rules
    print("\n" + "="*50)
    print("IF-THEN Rules:")
    if_then_rules = extract_if_then_rules(tree_entropy)
    for i, rule in enumerate(if_then_rules, 1):
        print(f"\nRule {i}:")
        print(f"  IF {rule['conditions']}")
        print(f"  THEN {rule['prediction']} (confidence: {rule['confidence']:.1%}, samples: {rule['samples']})")
    
    # Save comparison results
    comparison = pd.DataFrame({
        "Model": ["Tree-Entropy", "Tree-Gini", "Naive Bayes"],
        "Train_Acc": [results_entropy["train"]["accuracy"], results_gini["train"]["accuracy"], results_nb["train"]["accuracy"]],
        "Val_Acc": [results_entropy["val"]["accuracy"], results_gini["val"]["accuracy"], results_nb["val"]["accuracy"]],
        "Test_Acc": [results_entropy["test"]["accuracy"], results_gini["test"]["accuracy"], results_nb["test"]["accuracy"]],
        "Test_AUC": [results_entropy["test"]["roc_auc"], results_gini["test"]["roc_auc"], results_nb["test"]["roc_auc"]],
        "Test_F1": [results_entropy["test"]["f1"], results_gini["test"]["f1"], results_nb["test"]["f1"]]
    })
    comparison.to_csv("outputs/model_comparison_panel.csv", index=False)
    print("\nSaved model comparison to outputs/model_comparison_panel.csv")
