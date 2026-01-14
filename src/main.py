# =============================================================================
# ĐỒ ÁN: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, 
#        sinh con của giới trẻ Việt Nam (18–35)
# =============================================================================
# File: main.py
# Mô tả: Pipeline chính - Chạy toàn bộ quy trình từ load dữ liệu đến đánh giá
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thêm thư mục src vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    create_combined_dataset, 
    load_marriage_data, 
    load_tfr_data,
    get_summary_by_region,
    get_summary_by_year
)
from preprocessing import split_data, FEATURES_REAL, TARGET_REAL
from decision_tree_model import DecisionTreeModel, train_and_compare_trees, plot_roc_comparison
from naive_bayes_model import NaiveBayesModel, compare_nb_models
from visualization import (
    plot_target_distribution, 
    plot_feature_distributions,
    plot_metrics_comparison,
    save_all_plots
)


def print_data_summary(df):
    """In tổng quan dữ liệu"""
    print(f"\nKích thước dữ liệu: {df.shape}")
    print(f"Số tỉnh/thành: {df['tinh_thanh'].nunique()}")
    print(f"Số năm: {df['nam'].nunique()} ({df['nam'].min()} - {df['nam'].max()})")
    print(f"\nPhân bố theo vùng miền:")
    print(df.groupby('vung_mien')['tinh_thanh'].nunique())
    print(f"\nThống kê số liệu:")
    print(df[['so_ket_hon', 'tfr', 'ty_le_ket_hon']].describe())


def main():
    """
    Pipeline chính cho đồ án - Sử dụng dữ liệu thực từ nso
    """
    print("="*70)
    print("ĐỒ ÁN: PHÂN TÍCH XU HƯỚNG KẾT HÔN, SINH CON CỦA GIỚI TRẺ VIỆT NAM")
    print("Sử dụng: Decision Tree & Naive Bayes")
    print("Dữ liệu: Tổng cục Thống kê (nso) 2019-2024")
    print("="*70)
    
    # =========================================================================
    # BƯỚC 1: LOAD DỮ LIỆU THỰC
    # =========================================================================
    print("\n" + "-"*70)
    print("BƯỚC 1: LOAD DỮ LIỆU THỰC TỪ nso")
    print("-"*70)
    
    df = create_combined_dataset('data')
    print_data_summary(df)
    
    # Lưu dữ liệu đã xử lý
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/combined_data.csv", index=False, encoding="utf-8-sig")
    print("\n✓ Đã lưu dữ liệu vào: data/combined_data.csv")
    
    # In tổng hợp theo vùng và năm
    print("\n--- Tổng hợp theo vùng miền ---")
    summary_region = get_summary_by_region(df)
    print(summary_region.to_string())
    
    print("\n--- Tổng hợp theo năm ---")
    summary_year = get_summary_by_year(df)
    print(summary_year.to_string())
    
    # =========================================================================
    # BƯỚC 2: CHIA DỮ LIỆU
    # =========================================================================
    print("\n" + "-"*70)
    print("BƯỚC 2: CHIA DỮ LIỆU TRAIN/TEST (75/25)")
    print("-"*70)
    
    X_train, X_test, y_train, y_test = split_data(
        df, 
        features=FEATURES_REAL, 
        target=TARGET_REAL,
        test_size=0.25, 
        random_state=2026
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"\nTrain target distribution:")
    print(y_train.value_counts())
    print(f"\nTest target distribution:")
    print(y_test.value_counts())
    
    # =========================================================================
    # BƯỚC 3: HUẤN LUYỆN MÔ HÌNH DECISION TREE
    # =========================================================================
    print("\n" + "-"*70)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH DECISION TREE")
    print("-"*70)
    
    tree_results = train_and_compare_trees(
        X_train, X_test, y_train, y_test, 
        features=FEATURES_REAL
    )
    
    # In luật IF-THEN
    print("\n" + "="*50)
    print("LUẬT IF-THEN (Decision Tree - Entropy)")
    print("="*50)
    print(tree_results["entropy"]["model"].export_rules()[:2000])
    
    # =========================================================================
    # BƯỚC 4: HUẤN LUYỆN MÔ HÌNH NAIVE BAYES
    # =========================================================================
    print("\n" + "-"*70)
    print("BƯỚC 4: HUẤN LUYỆN MÔ HÌNH NAIVE BAYES")
    print("-"*70)
    
    nb_results = compare_nb_models(
        X_train, X_test, y_train, y_test,
        features=FEATURES_REAL
    )
    
    # Chọn model NB tốt nhất theo F1-score
    best_nb_type = max(nb_results.keys(), key=lambda k: nb_results[k]['metrics']['f1_score'])
    
    nb_model = nb_results[best_nb_type]['model']
    nb_metrics = nb_results[best_nb_type]['metrics']
    nb_threshold = nb_results[best_nb_type].get('best_threshold', 0.5)
    
    print(f"\n✓ Best Naive Bayes model: {best_nb_type} (threshold={nb_threshold:.2f})")
    
    # =========================================================================
    # BƯỚC 5: DỰ ĐOÁN CHO NĂM TIẾP THEO
    # =========================================================================
    print("\n" + "-"*70)
    print("BƯỚC 5: DỰ ĐOÁN XU HƯỚNG CHO NĂM 2025")
    print("-"*70)
    
    # Tạo dữ liệu dự đoán cho năm 2025 (giữ năm 2024 vì model chưa biết 2025)
    # Dự đoán dựa trên các đặc điểm của năm 2024
    df_2024 = df[df['nam'] == 2024].copy()
    df_2025_pred = df_2024[FEATURES_REAL].copy()
    # Giữ nguyên năm 2024 để model có thể dự đoán (vì 2025 không có trong training data)
    
    # Dự đoán
    pred_proba_tree = tree_results["entropy"]["model"].predict_proba(df_2025_pred)[:, 1]
    pred_proba_nb = nb_model.predict_proba(df_2025_pred)[:, 1]
    
    df_predictions = df_2024[['tinh_thanh', 'vung_mien', 'so_ket_hon']].copy()
    df_predictions['P_trend_up_tree'] = pred_proba_tree
    df_predictions['P_trend_up_nb'] = pred_proba_nb
    df_predictions['avg_probability'] = (pred_proba_tree + pred_proba_nb) / 2
    df_predictions['predicted_trend'] = (df_predictions['avg_probability'] >= 0.5).map({True: 'Tăng', False: 'Giảm'})
    
    print("\nDự đoán xu hướng kết hôn năm 2025 (dựa trên đặc điểm năm 2024):")
    print(df_predictions.to_string())
    
    # Tổng hợp theo vùng
    print("\n--- Tổng hợp dự đoán theo vùng miền ---")
    pred_by_region = df_predictions.groupby('vung_mien').agg({
        'P_trend_up_tree': 'mean',
        'P_trend_up_nb': 'mean',
        'avg_probability': 'mean'
    }).round(4)
    print(pred_by_region)
    
    # =========================================================================
    # BƯỚC 6: SO SÁNH CÁC MÔ HÌNH
    # =========================================================================
    print("\n" + "-"*70)
    print("BƯỚC 6: SO SÁNH CÁC MÔ HÌNH")
    print("-"*70)
    
    metrics_comparison = pd.DataFrame({
        "Tree-Entropy": {
            "Accuracy": tree_results["entropy"]["metrics"]["accuracy"],
            "Precision": tree_results["entropy"]["metrics"]["precision"],
            "Recall": tree_results["entropy"]["metrics"]["recall"],
            "F1-Score": tree_results["entropy"]["metrics"]["f1_score"],
            "ROC-AUC": tree_results["entropy"]["metrics"]["roc_auc"]
        },
        "Tree-Gini": {
            "Accuracy": tree_results["gini"]["metrics"]["accuracy"],
            "Precision": tree_results["gini"]["metrics"]["precision"],
            "Recall": tree_results["gini"]["metrics"]["recall"],
            "F1-Score": tree_results["gini"]["metrics"]["f1_score"],
            "ROC-AUC": tree_results["gini"]["metrics"]["roc_auc"]
        },
        f"NB-{best_nb_type}": {
            "Accuracy": nb_metrics["accuracy"],
            "Precision": nb_metrics["precision"],
            "Recall": nb_metrics["recall"],
            "F1-Score": nb_metrics["f1_score"],
            "ROC-AUC": nb_metrics["roc_auc"]
        }
    })
    
    print("\nBẢNG SO SÁNH CÁC MÔ HÌNH:")
    print(metrics_comparison.round(4).to_string())
    
    # Tìm model tốt nhất
    best_model = metrics_comparison.loc["Accuracy"].idxmax()
    print(f"\n✓ Best model (by Accuracy): {best_model}")
    
    # =========================================================================
    # BƯỚC 7: LƯU KẾT QUẢ
    # =========================================================================
    print("\n" + "-"*70)
    print("BƯỚC 7: LƯU KẾT QUẢ")
    print("-"*70)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Lưu models
    tree_results["entropy"]["model"].save_model("models/decision_tree_entropy.pkl")
    tree_results["gini"]["model"].save_model("models/decision_tree_gini.pkl")
    nb_model.save_model(f"models/naive_bayes_{best_nb_type}.pkl")
    
    # Lưu kết quả
    metrics_comparison.to_csv("outputs/model_comparison.csv")
    df_predictions.to_csv("outputs/predictions_2025.csv", index=False, encoding="utf-8-sig")
    
    print("✓ Đã lưu models vào thư mục: models/")
    print("✓ Đã lưu kết quả vào thư mục: outputs/")
    
    # =========================================================================
    # KẾT LUẬN
    # =========================================================================
    print("\n" + "="*70)
    print("KẾT LUẬN")
    print("="*70)
    
    print(f"""
    1. DỮ LIỆU:
       - Nguồn: Tổng cục Thống kê (nso) 2019-2024
       - Số tỉnh/thành: {df['tinh_thanh'].nunique()}
       - Giai đoạn: 2019-2024
       
    2. MÔ HÌNH:
       - Decision Tree (Entropy): Accuracy = {tree_results['entropy']['metrics']['accuracy']:.4f}
       - Decision Tree (Gini): Accuracy = {tree_results['gini']['metrics']['accuracy']:.4f}
       - Naive Bayes ({best_nb_type}): Accuracy = {nb_metrics['accuracy']:.4f}
       - Best model: {best_model}
       
    3. DỰ ĐOÁN 2025:
       - Số tỉnh dự đoán xu hướng TĂNG: {(df_predictions['predicted_trend'] == 'Tăng').sum()}
       - Số tỉnh dự đoán xu hướng GIẢM: {(df_predictions['predicted_trend'] == 'Giảm').sum()}
    """)
    
    print("\n" + "="*70)
    print("HOÀN THÀNH!")
    print("="*70)
    
    return {
        'df': df,
        'tree_results': tree_results,
        'nb_results': nb_results,
        'predictions': df_predictions,
        'metrics_comparison': metrics_comparison
    }


if __name__ == "__main__":
    results = main()
