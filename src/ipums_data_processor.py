# =============================================================================
# XỬ LÝ DỮ LIỆU IPUMS INTERNATIONAL - VIETNAM CENSUS DATA
# =============================================================================
# File: ipums_data_processor.py
# Mô tả: Xử lý dữ liệu điều tra dân số Việt Nam từ IPUMS International
#        để tạo dataset dự báo xác suất kết hôn
# Nguồn dữ liệu: IPUMS International (Minnesota Population Center)
#        - Vietnam 2009 Census
#        - Vietnam 2019 Census
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# ĐỊNH NGHĨA BIẾN - IPUMS CODEBOOK
# =============================================================================

# MARST - Marital Status
# 0 = Single/never married (Độc thân/chưa từng kết hôn)
# 1 = Married/in union (Đã kết hôn/sống chung)

# SEX - Sex
# 1 = Male (Nam)
# 2 = Female (Nữ)

# EDATTAIN - Educational Attainment
# 1 = Less than primary completed (Chưa hoàn thành tiểu học)
# 2 = Primary completed (Tiểu học)
# 3 = Secondary completed (THCS/THPT)
# 4 = University completed (ĐH/CĐ trở lên)

# URBAN - Urban/Rural
# 1 = Urban (Thành thị)
# 2 = Rural (Nông thôn)

# OWNERSHIP - Ownership of dwelling
# 1 = Owned (Sở hữu)
# 2 = Not owned (Không sở hữu)

# GEO1_VN - Region (First administrative level)
# 704001-704063 = Các tỉnh/thành phố Việt Nam

# =============================================================================
# MAPPING VÙNG MIỀN - BASED ON GEO1_VN CODES
# =============================================================================
# Mapping mã GEO1_VN sang 3 vùng chính
REGION_MAPPING = {
    # MIỀN BẮC (Bắc)
    704001: 'Bắc',  # Hà Nội
    704002: 'Bắc',  # Hà Giang
    704003: 'Bắc',  # Cao Bằng
    704004: 'Bắc',  # Bắc Kạn
    704005: 'Bắc',  # Tuyên Quang
    704006: 'Bắc',  # Lào Cai
    704007: 'Bắc',  # Điện Biên
    704008: 'Bắc',  # Lai Châu
    704009: 'Bắc',  # Sơn La
    704010: 'Bắc',  # Yên Bái
    704011: 'Bắc',  # Hòa Bình
    704012: 'Bắc',  # Thái Nguyên
    704013: 'Bắc',  # Lạng Sơn
    704014: 'Bắc',  # Quảng Ninh
    704015: 'Bắc',  # Bắc Giang
    704016: 'Bắc',  # Phú Thọ
    704017: 'Bắc',  # Vĩnh Phúc
    704018: 'Bắc',  # Bắc Ninh
    704019: 'Bắc',  # Hải Dương
    704020: 'Bắc',  # Hải Phòng
    704021: 'Bắc',  # Hưng Yên
    704022: 'Bắc',  # Thái Bình
    704023: 'Bắc',  # Hà Nam
    704024: 'Bắc',  # Nam Định
    704025: 'Bắc',  # Ninh Bình
    
    # MIỀN TRUNG (Trung)
    704026: 'Trung',  # Thanh Hóa
    704027: 'Trung',  # Nghệ An
    704028: 'Trung',  # Hà Tĩnh
    704029: 'Trung',  # Quảng Bình
    704030: 'Trung',  # Quảng Trị
    704031: 'Trung',  # Thừa Thiên Huế
    704032: 'Trung',  # Đà Nẵng
    704033: 'Trung',  # Quảng Nam
    704034: 'Trung',  # Quảng Ngãi
    704035: 'Trung',  # Bình Định
    704036: 'Trung',  # Phú Yên
    704037: 'Trung',  # Khánh Hòa
    704038: 'Trung',  # Ninh Thuận
    704039: 'Trung',  # Bình Thuận
    704040: 'Trung',  # Kon Tum
    704041: 'Trung',  # Gia Lai
    704042: 'Trung',  # Đắk Lắk
    704043: 'Trung',  # Đắk Nông
    704044: 'Trung',  # Lâm Đồng
    
    # MIỀN NAM (Nam)
    704045: 'Nam',  # Bình Phước
    704046: 'Nam',  # Tây Ninh
    704047: 'Nam',  # Bình Dương
    704048: 'Nam',  # Đồng Nai
    704049: 'Nam',  # Bà Rịa - Vũng Tàu
    704050: 'Nam',  # TP. Hồ Chí Minh
    704051: 'Nam',  # Long An
    704052: 'Nam',  # Tiền Giang
    704053: 'Nam',  # Bến Tre
    704054: 'Nam',  # Trà Vinh
    704055: 'Nam',  # Vĩnh Long
    704056: 'Nam',  # Đồng Tháp
    704057: 'Nam',  # An Giang
    704058: 'Nam',  # Kiên Giang
    704059: 'Nam',  # Cần Thơ
    704060: 'Nam',  # Hậu Giang
    704061: 'Nam',  # Sóc Trăng
    704062: 'Nam',  # Bạc Liêu
    704063: 'Nam',  # Cà Mau
}

# Thêm các mã mới (có thể từ census 2019)
for code in range(704064, 704100):
    if code not in REGION_MAPPING:
        # Mặc định gán theo pattern
        if code <= 704075:
            REGION_MAPPING[code] = 'Trung'
        elif code <= 704090:
            REGION_MAPPING[code] = 'Nam'
        else:
            REGION_MAPPING[code] = 'Bắc'


def load_ipums_data(file_path: str = "data/ipumsi_data.csv", 
                    sample_size: int = None,
                    age_range: tuple = (18, 35)) -> pd.DataFrame:
    """
    Load và tiền xử lý dữ liệu IPUMS International Vietnam
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file CSV
    sample_size : int, optional
        Số dòng sample (None = load toàn bộ)
    age_range : tuple
        Khoảng tuổi cần lọc (min, max)
        
    Returns:
    --------
    pd.DataFrame : Dữ liệu đã được xử lý
    """
    print(f"Loading IPUMS data from {file_path}...")
    
    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)
    
    print(f"  Loaded {len(df):,} records")
    
    # Lọc theo độ tuổi
    df = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]
    print(f"  After age filter ({age_range[0]}-{age_range[1]}): {len(df):,} records")
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các biến features từ dữ liệu IPUMS
    
    Biến đầu vào (IPUMS):
    - YEAR: Năm điều tra
    - AGE: Tuổi
    - SEX: Giới tính (1=Nam, 2=Nữ)
    - EDATTAIN: Trình độ học vấn (1-4)
    - URBAN: Thành thị/Nông thôn (1=Urban, 2=Rural)
    - OWNERSHIP: Sở hữu nhà (1=Owned, 2=Not owned)
    - GEO1_VN: Mã vùng địa lý
    - LIVEAREA: Diện tích nhà ở (m2)
    - MARST: Tình trạng hôn nhân (0=Single, 1=Married)
    
    Biến đầu ra (Features):
    - year: Năm
    - age: Tuổi
    - age_group: Nhóm tuổi (18-24, 25-29, 30-35)
    - sex: Giới tính (Nam, Nữ)
    - education: Trình độ học vấn
    - urban_rural: Thành thị/Nông thôn
    - region: Vùng miền (Bắc, Trung, Nam)
    - home_ownership: Sở hữu nhà (0/1)
    - living_area_level: Mức diện tích nhà ở
    - Y_married: Biến mục tiêu - đã kết hôn (0/1)
    """
    print("Creating features...")
    
    # Copy dataframe
    data = df.copy()
    
    # ----- BIẾN NĂM -----
    data['year'] = data['YEAR']
    
    # ----- BIẾN TUỔI -----
    data['age'] = data['AGE']
    
    # Nhóm tuổi
    data['age_group'] = pd.cut(
        data['AGE'], 
        bins=[17, 24, 29, 35], 
        labels=['18-24', '25-29', '30-35']
    )
    
    # ----- BIẾN GIỚI TÍNH -----
    data['sex'] = data['SEX'].map({1: 'Nam', 2: 'Nữ'})
    
    # ----- BIẾN HỌC VẤN -----
    # EDATTAIN: 1=Less than primary, 2=Primary, 3=Secondary, 4=University
    education_map = {
        1: '≤Tiểu học',      # Less than primary completed
        2: 'THCS',           # Primary completed  
        3: 'THPT',           # Secondary completed
        4: 'ĐH/CĐ+'          # University completed
    }
    data['education'] = data['EDATTAIN'].map(education_map)
    
    # Nhóm học vấn đơn giản hơn cho model
    education_simple_map = {
        1: '≤THPT',
        2: '≤THPT', 
        3: '≤THPT',
        4: 'ĐH/CĐ+'
    }
    data['education_level'] = data['EDATTAIN'].map(education_simple_map)
    
    # ----- BIẾN THÀNH THỊ/NÔNG THÔN -----
    data['urban_rural'] = data['URBAN'].map({1: 'Đô thị', 2: 'Nông thôn'})
    
    # ----- BIẾN VÙNG MIỀN -----
    data['region'] = data['GEO1_VN'].map(REGION_MAPPING)
    # Xử lý các mã không có trong mapping
    data['region'] = data['region'].fillna('Khác')
    
    # ----- BIẾN SỞ HỮU NHÀ -----
    data['home_ownership'] = data['OWNERSHIP'].map({1: 1, 2: 0})
    
    # ----- BIẾN DIỆN TÍCH NHÀ Ở -----
    # Chia thành các mức dựa trên phân vị
    data['living_area'] = data['LIVEAREA']
    
    # Quartile-based levels
    q25, q50, q75 = data['LIVEAREA'].quantile([0.25, 0.5, 0.75])
    data['living_area_level'] = pd.cut(
        data['LIVEAREA'],
        bins=[0, q25, q50, q75, float('inf')],
        labels=['Nhỏ', 'Trung bình', 'Khá', 'Rộng']
    )
    
    # ----- BIẾN MỤC TIÊU -----
    data['Y_married'] = data['MARST']  # 0=Single, 1=Married
    
    # ----- BIẾN BỔ SUNG (TỪ SERIAL/HOUSEHOLD) -----
    # Số người trong hộ gia đình
    household_size = data.groupby(['YEAR', 'SERIAL']).size().reset_index(name='household_size')
    data = data.merge(household_size, on=['YEAR', 'SERIAL'], how='left')
    
    # Nhóm quy mô hộ
    data['household_size_group'] = pd.cut(
        data['household_size'],
        bins=[0, 2, 4, 6, float('inf')],
        labels=['1-2 người', '3-4 người', '5-6 người', '>6 người']
    )
    
    print(f"  Created {len(data.columns)} columns")
    
    return data


def create_panel_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo dataset cho phân tích panel từ dữ liệu 2 năm (2009, 2019)
    
    Returns:
    --------
    pd.DataFrame : Panel dataset với các biến đã chuẩn hóa
    """
    print("Creating panel dataset...")
    
    # Chọn các cột cần thiết cho model
    feature_cols = [
        'year',
        'age',
        'age_group',
        'sex',
        'education',
        'education_level',
        'urban_rural',
        'region',
        'home_ownership',
        'living_area',
        'living_area_level',
        'household_size',
        'household_size_group',
        'Y_married'
    ]
    
    # Lọc cột tồn tại
    existing_cols = [col for col in feature_cols if col in df.columns]
    panel_data = df[existing_cols].copy()
    
    # Xóa missing values
    panel_data = panel_data.dropna()
    
    print(f"  Panel dataset: {len(panel_data):,} records")
    print(f"  Years: {panel_data['year'].unique()}")
    print(f"  Marriage rate: {panel_data['Y_married'].mean():.2%}")
    
    return panel_data


def get_feature_summary(df: pd.DataFrame) -> dict:
    """
    Tạo summary thống kê các features
    """
    summary = {
        'total_records': len(df),
        'years': df['year'].unique().tolist(),
        'marriage_rate': df['Y_married'].mean(),
        
        'by_year': df.groupby('year')['Y_married'].agg(['count', 'mean']).to_dict(),
        'by_age_group': df.groupby('age_group')['Y_married'].agg(['count', 'mean']).to_dict(),
        'by_sex': df.groupby('sex')['Y_married'].agg(['count', 'mean']).to_dict(),
        'by_education': df.groupby('education_level')['Y_married'].agg(['count', 'mean']).to_dict(),
        'by_urban_rural': df.groupby('urban_rural')['Y_married'].agg(['count', 'mean']).to_dict(),
        'by_region': df.groupby('region')['Y_married'].agg(['count', 'mean']).to_dict(),
        'by_home_ownership': df.groupby('home_ownership')['Y_married'].agg(['count', 'mean']).to_dict(),
    }
    
    return summary


def save_processed_data(df: pd.DataFrame, output_path: str = "data/ipums_processed.csv"):
    """Lưu dữ liệu đã xử lý"""
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


# =============================================================================
# MAIN PROCESSING
# =============================================================================
if __name__ == "__main__":
    # Load dữ liệu
    raw_data = load_ipums_data("data/ipumsi_data.csv", age_range=(18, 35))
    
    # Tạo features
    processed_data = create_features(raw_data)
    
    # Tạo panel dataset
    panel_data = create_panel_dataset(processed_data)
    
    # In summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal records: {len(panel_data):,}")
    print(f"Years: {panel_data['year'].unique()}")
    print(f"Overall marriage rate: {panel_data['Y_married'].mean():.2%}")
    
    print("\n--- Marriage rate by Age Group ---")
    print(panel_data.groupby('age_group')['Y_married'].agg(['count', 'mean']))
    
    print("\n--- Marriage rate by Sex ---")
    print(panel_data.groupby('sex')['Y_married'].agg(['count', 'mean']))
    
    print("\n--- Marriage rate by Education ---")
    print(panel_data.groupby('education_level')['Y_married'].agg(['count', 'mean']))
    
    print("\n--- Marriage rate by Urban/Rural ---")
    print(panel_data.groupby('urban_rural')['Y_married'].agg(['count', 'mean']))
    
    print("\n--- Marriage rate by Region ---")
    print(panel_data.groupby('region')['Y_married'].agg(['count', 'mean']))
    
    print("\n--- Marriage rate by Home Ownership ---")
    print(panel_data.groupby('home_ownership')['Y_married'].agg(['count', 'mean']))
    
    # Lưu dữ liệu
    save_processed_data(panel_data, "data/ipums_processed.csv")
    
    print("\nProcessing completed!")
