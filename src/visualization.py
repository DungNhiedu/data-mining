# =============================================================================
# ĐỒ ÁN: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, 
#        sinh con của giới trẻ Việt Nam (18–35)
# =============================================================================
# File: visualization.py
# Mô tả: Các hàm visualization cho phân tích và báo cáo
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix


# Cấu hình mặc định cho matplotlib với font hỗ trợ tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


def plot_target_distribution(df, target_col="marital_class"):
    """
    Vẽ biểu đồ phân phối biến target
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    counts = df[target_col].value_counts()
    labels = ["Chưa kết hôn (0)", "Đã kết hôn (1)"]
    colors = ["#3498db", "#e74c3c"]
    
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='white', linewidth=2)
    
    # Thêm giá trị trên cột
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                f'{count}\n({count/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=12)
    
    ax.set_title("Phân phối biến target (marital_class)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Số lượng", fontsize=12)
    ax.set_xlabel("Tình trạng hôn nhân", fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_feature_distributions(df):
    """
    Vẽ biểu đồ phân phối các features
    """
    categorical_cols = ["age_group", "sex", "region", "urban_rural", "education", "income_level"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        
        # Đếm theo biến target
        cross_tab = pd.crosstab(df[col], df["marital_class"])
        cross_tab.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], edgecolor='white')
        
        ax.set_title(f"Phân phối {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Số lượng", fontsize=10)
        ax.legend(["Chưa kết hôn", "Đã kết hôn"], loc='upper right', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Phân phối các features theo tình trạng hôn nhân", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_marriage_rate_by_features(df):
    """
    Vẽ tỷ lệ kết hôn theo các features
    """
    categorical_cols = ["age_group", "sex", "region", "urban_rural", "education", "income_level"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        
        # Tính tỷ lệ kết hôn theo từng category
        marriage_rate = df.groupby(col)["marital_class"].mean()
        
        bars = ax.bar(marriage_rate.index, marriage_rate.values * 100, 
                     color='#27ae60', edgecolor='white')
        
        # Thêm giá trị trên cột
        for bar, rate in zip(bars, marriage_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f"Tỷ lệ kết hôn theo {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Tỷ lệ kết hôn (%)", fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 100)
    
    plt.suptitle("Tỷ lệ kết hôn theo các đặc điểm nhân khẩu học", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_child_rate_by_education_urban(df):
    """
    Vẽ biểu đồ tỷ lệ có con theo education x urban_rural
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df.pivot_table(
        values='has_child', 
        index='education', 
        columns='urban_rural', 
        aggfunc='mean'
    ) * 100
    
    pivot.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], edgecolor='white')
    
    ax.set_title("Tỷ lệ có con theo Học vấn và Khu vực", fontsize=14, fontweight='bold')
    ax.set_xlabel("Học vấn", fontsize=12)
    ax.set_ylabel("Tỷ lệ có con (%)", fontsize=12)
    ax.legend(title="Khu vực", fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig


def plot_income_distribution(df):
    """
    Vẽ phân phối thu nhập
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram thu nhập
    ax1 = axes[0]
    df['income_pc'].hist(bins=50, ax=ax1, color='#9b59b6', edgecolor='white', alpha=0.7)
    ax1.set_title("Phân phối Thu nhập bình quân đầu người", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Thu nhập (nghìn VNĐ/tháng)", fontsize=10)
    ax1.set_ylabel("Tần suất", fontsize=10)
    
    # Boxplot theo marital_class
    ax2 = axes[1]
    df.boxplot(column='income_pc', by='marital_class', ax=ax2)
    ax2.set_title("Thu nhập theo tình trạng hôn nhân", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Tình trạng hôn nhân (0: Chưa, 1: Đã)", fontsize=10)
    ax2.set_ylabel("Thu nhập", fontsize=10)
    plt.suptitle("")  # Bỏ title mặc định
    
    plt.tight_layout()
    return fig


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Vẽ ROC curves so sánh các mô hình
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary {model_name: model_pipeline}
    X_test : pd.DataFrame
        Features test set
    y_test : pd.Series
        Target test set
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#3498db', '#e74c3c', '#27ae60']
    
    for idx, (name, model) in enumerate(models_dict.items()):
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, 
            name=name, 
            ax=ax,
            color=colors[idx % len(colors)]
        )
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_title("So sánh ROC Curves của các mô hình", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrices(results_dict, figsize=(15, 5)):
    """
    Vẽ confusion matrices của nhiều mô hình
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary {model_name: confusion_matrix}
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    cmaps = ['Blues', 'Reds', 'Greens']
    
    for idx, (name, cm) in enumerate(results_dict.items()):
        ax = axes[idx]
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap=cmaps[idx % len(cmaps)],
            xticklabels=["Chưa KH", "Đã KH"],
            yticklabels=["Chưa KH", "Đã KH"],
            ax=ax
        )
        ax.set_title(f"Confusion Matrix\n{name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_df):
    """
    Vẽ biểu đồ so sánh các chỉ số đánh giá
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame với columns là các model, rows là các metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_df.plot(kind='bar', ax=ax, width=0.8, edgecolor='white')
    
    ax.set_title("So sánh các chỉ số đánh giá của mô hình", fontsize=14, fontweight='bold')
    ax.set_xlabel("Chỉ số đánh giá", fontsize=12)
    ax.set_ylabel("Giá trị", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(title="Mô hình", fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # Thêm đường tham chiếu
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def save_all_plots(df, output_dir="outputs/figures"):
    """
    Lưu tất cả biểu đồ vào thư mục
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu các biểu đồ
    plots = {
        "target_distribution.png": plot_target_distribution(df),
        "feature_distributions.png": plot_feature_distributions(df),
        "marriage_rate_by_features.png": plot_marriage_rate_by_features(df),
        "child_rate_education_urban.png": plot_child_rate_by_education_urban(df),
        "income_distribution.png": plot_income_distribution(df)
    }
    
    for filename, fig in plots.items():
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"[OK] Da luu: {filepath}")
        plt.close(fig)


if __name__ == "__main__":
    from data_generation import generate_synthetic_data
    
    # Sinh dữ liệu
    df = generate_synthetic_data(n_samples=3000)
    
    # Lưu tất cả biểu đồ
    save_all_plots(df)
    print("\n[OK] Hoan thanh tao bieu do!")
