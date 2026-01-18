# =============================================================================
# ĐỒ ÁN: Dự báo Xu hướng Kết hôn của Giới trẻ Việt Nam (18-35)
# =============================================================================
# File: data_processor.py
# Mô tả: Xử lý và chuẩn bị dữ liệu IPUMS International Vietnam Census
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Định nghĩa mapping cho các biến
URBAN_MAP = {1: 'Urban', 2: 'Rural'}
SEX_MAP = {1: 'Male', 2: 'Female'}
MARST_MAP = {0: 'Single', 1: 'Married'}
EDATTAIN_MAP = {
    1: 'Less_than_primary',
    2: 'Primary', 
    3: 'Secondary',
    4: 'University_plus'
}
OWNERSHIP_MAP = {1: 'Owned', 2: 'Not_owned'}

# Mapping vùng địa lý Việt Nam (GEO1_VN codes)
GEO1_VN_MAP = {
    704001: 'Ha_Noi',
    704002: 'Ha_Giang',
    704003: 'Cao_Bang', 
    704004: 'Bac_Kan',
    704005: 'Tuyen_Quang',
    704006: 'Lao_Cai',
    704007: 'Dien_Bien',
    704008: 'Lai_Chau',
    704009: 'Son_La',
    704010: 'Yen_Bai',
    704011: 'Hoa_Binh',
    704012: 'Thai_Nguyen',
    704013: 'Lang_Son',
    704014: 'Quang_Ninh',
    704015: 'Bac_Giang',
    704016: 'Phu_Tho',
    704017: 'Vinh_Phuc',
    704018: 'Bac_Ninh',
    704019: 'Hai_Duong',
    704020: 'Hai_Phong',
    704021: 'Hung_Yen',
    704022: 'Thai_Binh',
    704023: 'Ha_Nam',
    704024: 'Nam_Dinh',
    704025: 'Ninh_Binh',
    704026: 'Thanh_Hoa',
    704027: 'Nghe_An',
    704028: 'Ha_Tinh',
    704029: 'Quang_Binh',
    704030: 'Quang_Tri',
    704031: 'Thua_Thien_Hue',
    704032: 'Da_Nang',
    704033: 'Quang_Nam',
    704034: 'Quang_Ngai',
    704035: 'Binh_Dinh',
    704036: 'Phu_Yen',
    704037: 'Khanh_Hoa',
    704038: 'Ninh_Thuan',
    704039: 'Binh_Thuan',
    704040: 'Kon_Tum',
    704041: 'Gia_Lai',
    704042: 'Dak_Lak',
    704043: 'Dak_Nong',
    704044: 'Lam_Dong',
    704045: 'Binh_Phuoc',
    704046: 'Tay_Ninh',
    704047: 'Binh_Duong',
    704048: 'Dong_Nai',
    704049: 'Ba_Ria_Vung_Tau',
    704050: 'Ho_Chi_Minh',
    704051: 'Long_An',
    704052: 'Tien_Giang',
    704053: 'Ben_Tre',
    704054: 'Tra_Vinh',
    704055: 'Vinh_Long',
    704056: 'Dong_Thap',
    704057: 'An_Giang',
    704058: 'Kien_Giang',
    704059: 'Can_Tho',
    704060: 'Hau_Giang',
    704061: 'Soc_Trang',
    704062: 'Bac_Lieu',
    704063: 'Ca_Mau',
}

# Phân vùng địa lý
REGION_NORTH = [704001, 704002, 704003, 704004, 704005, 704006, 704007, 704008, 
                704009, 704010, 704011, 704012, 704013, 704014, 704015, 704016,
                704017, 704018, 704019, 704020, 704021, 704022, 704023, 704024, 704025]
REGION_CENTRAL = [704026, 704027, 704028, 704029, 704030, 704031, 704032, 704033,
                  704034, 704035, 704036, 704037, 704038, 704039, 704040, 704041,
                  704042, 704043, 704044]
REGION_SOUTH = [704045, 704046, 704047, 704048, 704049, 704050, 704051, 704052,
                704053, 704054, 704055, 704056, 704057, 704058, 704059, 704060,
                704061, 704062, 704063]


def get_region(geo_code):
    """Xác định vùng từ mã địa lý"""
    if geo_code in REGION_NORTH:
        return 'North'
    elif geo_code in REGION_CENTRAL:
        return 'Central'
    elif geo_code in REGION_SOUTH:
        return 'South'
    else:
        return 'Unknown'


def get_age_group(age):
    """Phân nhóm tuổi"""
    if 18 <= age <= 24:
        return '18-24'
    elif 25 <= age <= 29:
        return '25-29'
    elif 30 <= age <= 35:
        return '30-35'
    else:
        return 'Other'


def load_and_process_data(filepath='data/ipumsi_data.csv', sample_size=None):
    """
    Tải và xử lý dữ liệu IPUMS
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn đến file CSV
    sample_size : int, optional
        Số lượng mẫu cần lấy (None = lấy tất cả)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame đã được xử lý
    """
    print("Đang tải dữ liệu...")
    
    if sample_size:
        df = pd.read_csv(filepath, nrows=sample_size)
    else:
        df = pd.read_csv(filepath)
    
    print(f"Đã tải {len(df):,} bản ghi")
    
    # Lọc độ tuổi 18-35 (giới trẻ)
    df = df[(df['AGE'] >= 18) & (df['AGE'] <= 35)].copy()
    print(f"Sau khi lọc độ tuổi 18-35: {len(df):,} bản ghi")
    
    # Tạo các biến mới
    print("Đang xử lý dữ liệu...")
    
    # Mapping các biến phân loại
    df['urban_label'] = df['URBAN'].map(URBAN_MAP)
    df['sex_label'] = df['SEX'].map(SEX_MAP)
    df['marst_label'] = df['MARST'].map(MARST_MAP)
    df['edattain_label'] = df['EDATTAIN'].map(EDATTAIN_MAP)
    df['ownership_label'] = df['OWNERSHIP'].map(OWNERSHIP_MAP)
    
    # Tạo biến vùng
    df['region'] = df['GEO1_VN'].apply(get_region)
    
    # Tạo nhóm tuổi
    df['age_group'] = df['AGE'].apply(get_age_group)
    
    # Tạo biến target (đã kết hôn = 1, chưa = 0)
    df['is_married'] = df['MARST']
    
    # Loại bỏ các dòng có giá trị thiếu trong các biến quan trọng
    important_cols = ['YEAR', 'URBAN', 'SEX', 'AGE', 'EDATTAIN', 'MARST', 'OWNERSHIP']
    df = df.dropna(subset=important_cols)
    
    print(f"Xử lý hoàn tất: {len(df):,} bản ghi")
    
    return df


def prepare_features(df):
    """
    Chuẩn bị features cho mô hình
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đã xử lý
    
    Returns:
    --------
    tuple
        (X, y, feature_names, encoders)
    """
    # Các features sử dụng
    feature_cols = ['YEAR', 'URBAN', 'SEX', 'AGE', 'EDATTAIN', 'OWNERSHIP', 'region']
    
    # Tạo copy để tránh warning
    df_features = df[feature_cols + ['is_married']].copy()
    
    # Encode biến region
    le_region = LabelEncoder()
    df_features['region_encoded'] = le_region.fit_transform(df_features['region'])
    
    # Tạo features cuối cùng
    X = df_features[['YEAR', 'URBAN', 'SEX', 'AGE', 'EDATTAIN', 'OWNERSHIP', 'region_encoded']].values
    y = df_features['is_married'].values
    
    feature_names = ['YEAR', 'URBAN', 'SEX', 'AGE', 'EDATTAIN', 'OWNERSHIP', 'REGION']
    
    encoders = {'region': le_region}
    
    return X, y, feature_names, encoders


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành tập train và test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_summary_statistics(df):
    """
    Tính toán thống kê tổng quan
    """
    stats = {
        'total_records': len(df),
        'years': sorted(df['YEAR'].unique().tolist()),
        'marriage_rate': df['is_married'].mean() * 100,
        'by_year': df.groupby('YEAR')['is_married'].mean().to_dict(),
        'by_sex': df.groupby('sex_label')['is_married'].mean().to_dict(),
        'by_urban': df.groupby('urban_label')['is_married'].mean().to_dict(),
        'by_education': df.groupby('edattain_label')['is_married'].mean().to_dict(),
        'by_age_group': df.groupby('age_group')['is_married'].mean().to_dict(),
        'by_region': df.groupby('region')['is_married'].mean().to_dict(),
    }
    return stats


if __name__ == "__main__":
    # Test
    df = load_and_process_data(sample_size=100000)
    print("\nThống kê tổng quan:")
    stats = get_summary_statistics(df)
    for key, value in stats.items():
        print(f"  {key}: {value}")
