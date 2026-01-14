# =============================================================================
# ĐỒ ÁN: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, 
#        sinh con của giới trẻ Việt Nam (18–35)
# =============================================================================
# File: preprocessing.py
# Mô tả: Tiền xử lý dữ liệu - Mã hóa, chia train/test
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Định nghĩa các features cho dữ liệu thực (nso data)
# Features cho mô hình dự đoán xu hướng kết hôn
FEATURES_REAL = ['vung_mien', 'vung_kinh_te', 'urban_rural', 'tfr_level', 'nam']
FEATURES_NUMERIC = ['tfr', 'ty_le_ket_hon', 'dan_so_2019', 'mat_do']
TARGET_REAL = 'trend'  # 0: Giảm, 1: Tăng/Giữ nguyên

# Features cũ cho synthetic data (giữ lại để tương thích)
FEATURES = ["age_group", "sex", "region", "urban_rural", "education", "income_level", "has_child"]
TARGET = "marital_class"


def load_data(filepath):
    """
    Đọc dữ liệu từ file CSV
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn đến file CSV
        
    Returns:
    --------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)
    return df


def load_real_data(data_dir='data'):
    """
    Đọc dữ liệu thực từ các file Excel và tạo dataset tổng hợp
    
    Parameters:
    -----------
    data_dir : str
        Thư mục chứa các file dữ liệu
        
    Returns:
    --------
    pd.DataFrame
    """
    from data_loader import create_combined_dataset
    return create_combined_dataset(data_dir)


def prepare_ml_data(df, features=None, target=None, drop_na=True):
    """
    Chuẩn bị dữ liệu cho Machine Learning
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    features : list
        Danh sách features (mặc định sử dụng FEATURES_REAL)
    target : str
        Tên cột target (mặc định sử dụng TARGET_REAL)
    drop_na : bool
        Loại bỏ các dòng có giá trị NaN
        
    Returns:
    --------
    tuple: (X, y, feature_names)
    """
    if features is None:
        features = FEATURES_REAL
    if target is None:
        target = TARGET_REAL
    
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna(subset=features + [target])
    
    X = df_clean[features]
    y = df_clean[target]
    
    return X, y, features


def split_data(df, features=None, target=None, test_size=0.25, random_state=2026):
    """
    Chia dữ liệu thành tập train và test
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    features : list
        Danh sách features (nếu None, sử dụng FEATURES_REAL)
    target : str
        Tên cột target (nếu None, sử dụng TARGET_REAL)
    test_size : float
        Tỷ lệ tập test (mặc định 0.25)
    random_state : int
        Seed để tái tạo kết quả
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if features is None:
        features = FEATURES_REAL
    if target is None:
        target = TARGET_REAL
    
    # Loại bỏ NaN
    df_clean = df.dropna(subset=features + [target])
    
    X = df_clean[features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def split_data_legacy(df, test_size=0.25, random_state=2026):
    """
    Chia dữ liệu (phiên bản cũ cho synthetic data)
    """
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def get_tree_preprocessor(features=None):
    """
    Tạo preprocessor cho Decision Tree (One-Hot Encoding)
    
    Parameters:
    -----------
    features : list
        Danh sách features cần encode
    
    Returns:
    --------
    ColumnTransformer
    """
    if features is None:
        features = FEATURES_REAL
    
    preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features)
    ], remainder="drop")
    
    return preprocessor


def get_nb_preprocessor(features=None):
    """
    Tạo preprocessor cho Naive Bayes (Ordinal Encoding)
    
    Parameters:
    -----------
    features : list
        Danh sách features cần encode
    
    Returns:
    --------
    ColumnTransformer
    """
    if features is None:
        features = FEATURES_REAL
    
    preprocessor = ColumnTransformer([
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), features)
    ], remainder="drop")
    
    return preprocessor


def get_mixed_preprocessor(categorical_features, numeric_features=None):
    """
    Tạo preprocessor cho dữ liệu hỗn hợp (categorical + numeric)
    
    Parameters:
    -----------
    categorical_features : list
        Danh sách features categorical
    numeric_features : list
        Danh sách features numeric
    
    Returns:
    --------
    ColumnTransformer
    """
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
    
    if numeric_features:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )
    
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    
    return preprocessor


def encode_target(y):
    """
    Encode target variable nếu cần
    
    Parameters:
    -----------
    y : pd.Series
        Target variable
        
    Returns:
    --------
    tuple: (y_encoded, label_encoder)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def get_feature_info():
    """
    Trả về thông tin về các features cho dữ liệu thực
    
    Returns:
    --------
    dict
    """
    return {
        "categorical_features": FEATURES_REAL,
        "numeric_features": FEATURES_NUMERIC,
        "target": TARGET_REAL,
        "description": {
            "vung_mien": "Vùng miền (Bắc, Trung, Nam)",
            "vung_kinh_te": "Vùng kinh tế (6 vùng)",
            "urban_rural": "Đô thị / Bán đô thị / Nông thôn",
            "tfr_level": "Mức tỷ suất sinh (Thấp, Trung bình, Cao)",
            "nam": "Năm (2019-2024)",
            "tfr": "Tổng tỷ suất sinh",
            "ty_le_ket_hon": "Tỷ lệ kết hôn trên 1000 dân",
            "dan_so_2019": "Dân số năm 2019",
            "mat_do": "Mật độ dân số",
            "trend": "Xu hướng kết hôn (0: Giảm, 1: Tăng/Giữ nguyên)"
        }
    }


def get_feature_info_legacy():
    """
    Trả về thông tin về các features (phiên bản cũ cho synthetic data)
    
    Returns:
    --------
    dict
    """
    return {
        "features": FEATURES,
        "target": TARGET,
        "description": {
            "age_group": "Nhóm tuổi (18-24, 25-29, 30-35)",
            "sex": "Giới tính (Nam, Nữ)",
            "region": "Vùng miền (Bắc, Trung, Nam)",
            "urban_rural": "Nơi sống (Đô thị, Nông thôn)",
            "education": "Học vấn (≤THPT, CĐ, ĐH, >ĐH)",
            "income_level": "Mức thu nhập (low, mid, high)",
            "has_child": "Có con hay chưa (0, 1)",
            "marital_class": "Tình trạng hôn nhân (0: chưa, 1: đã/đang)"
        }
    }


if __name__ == "__main__":
    # Test với dữ liệu thực
    print("="*60)
    print("TESTING PREPROCESSING WITH REAL DATA")
    print("="*60)
    
    # Load dữ liệu
    df = load_real_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Chuẩn bị dữ liệu ML
    X, y, features = prepare_ml_data(df)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Test preprocessors
    print("\nTesting Tree Preprocessor...")
    pre_tree = get_tree_preprocessor()
    X_tree = pre_tree.fit_transform(X_train)
    print(f"Transformed shape: {X_tree.shape}")
    
    print("\nTesting NB Preprocessor...")
    pre_nb = get_nb_preprocessor()
    X_nb = pre_nb.fit_transform(X_train)
    print(f"Transformed shape: {X_nb.shape}")
    
    print("\n✓ All preprocessing tests passed!")
