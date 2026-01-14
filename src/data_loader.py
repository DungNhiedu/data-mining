# =============================================================================
# ĐỒ ÁN: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, 
#        sinh con của giới trẻ Việt Nam (18–35)
# =============================================================================
# File: data_loader.py
# Mô tả: Đọc và xử lý dữ liệu thực từ các file Excel (nso data)
# =============================================================================

import pandas as pd
import numpy as np
import os

# Định nghĩa các vùng miền
VUNG_MIEN = {
    'Đồng bằng sông Hồng': 'Bắc',
    'Trung du và miền núi phía Bắc': 'Bắc',
    'Bắc Trung Bộ và Duyên hải miền Trung': 'Trung',
    'Tây Nguyên': 'Trung',
    'Đông Nam Bộ': 'Nam',
    'Đồng bằng sông Cửu Long': 'Nam'
}

# Mapping tỉnh thành theo vùng kinh tế
TINH_THEO_VUNG = {
    # Đồng bằng sông Hồng
    'Hà Nội': 'Đồng bằng sông Hồng',
    'Vĩnh Phúc': 'Đồng bằng sông Hồng',
    'Bắc Ninh': 'Đồng bằng sông Hồng',
    'Quảng Ninh': 'Đồng bằng sông Hồng',
    'Hải Dương': 'Đồng bằng sông Hồng',
    'Hải Phòng': 'Đồng bằng sông Hồng',
    'Hưng Yên': 'Đồng bằng sông Hồng',
    'Thái Bình': 'Đồng bằng sông Hồng',
    'Hà Nam': 'Đồng bằng sông Hồng',
    'Nam Định': 'Đồng bằng sông Hồng',
    'Ninh Bình': 'Đồng bằng sông Hồng',
    
    # Trung du và miền núi phía Bắc
    'Hà Giang': 'Trung du và miền núi phía Bắc',
    'Cao Bằng': 'Trung du và miền núi phía Bắc',
    'Bắc Kạn': 'Trung du và miền núi phía Bắc',
    'Tuyên Quang': 'Trung du và miền núi phía Bắc',
    'Lào Cai': 'Trung du và miền núi phía Bắc',
    'Yên Bái': 'Trung du và miền núi phía Bắc',
    'Thái Nguyên': 'Trung du và miền núi phía Bắc',
    'Lạng Sơn': 'Trung du và miền núi phía Bắc',
    'Bắc Giang': 'Trung du và miền núi phía Bắc',
    'Phú Thọ': 'Trung du và miền núi phía Bắc',
    'Điện Biên': 'Trung du và miền núi phía Bắc',
    'Lai Châu': 'Trung du và miền núi phía Bắc',
    'Sơn La': 'Trung du và miền núi phía Bắc',
    'Hoà Bình': 'Trung du và miền núi phía Bắc',
    'Hòa Bình': 'Trung du và miền núi phía Bắc',
    
    # Bắc Trung Bộ và Duyên hải miền Trung
    'Thanh Hoá': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Thanh Hóa': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Nghệ An': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Hà Tĩnh': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Quảng Bình': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Quảng Trị': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Thừa Thiên Huế': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Đà Nẵng': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Quảng Nam': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Quảng Ngãi': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Bình Định': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Phú Yên': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Khánh Hoà': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Khánh Hòa': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Ninh Thuận': 'Bắc Trung Bộ và Duyên hải miền Trung',
    'Bình Thuận': 'Bắc Trung Bộ và Duyên hải miền Trung',
    
    # Tây Nguyên
    'Kon Tum': 'Tây Nguyên',
    'Gia Lai': 'Tây Nguyên',
    'Đắk Lắk': 'Tây Nguyên',
    'Đắk Nông': 'Tây Nguyên',
    'Lâm Đồng': 'Tây Nguyên',
    
    # Đông Nam Bộ
    'Bình Phước': 'Đông Nam Bộ',
    'Tây Ninh': 'Đông Nam Bộ',
    'Bình Dương': 'Đông Nam Bộ',
    'Đồng Nai': 'Đông Nam Bộ',
    'Bà Rịa - Vũng Tàu': 'Đông Nam Bộ',
    'TP.Hồ Chí Minh': 'Đông Nam Bộ',
    'Hồ Chí Minh': 'Đông Nam Bộ',
    
    # Đồng bằng sông Cửu Long
    'Long An': 'Đồng bằng sông Cửu Long',
    'Tiền Giang': 'Đồng bằng sông Cửu Long',
    'Bến Tre': 'Đồng bằng sông Cửu Long',
    'Trà Vinh': 'Đồng bằng sông Cửu Long',
    'Vĩnh Long': 'Đồng bằng sông Cửu Long',
    'Đồng Tháp': 'Đồng bằng sông Cửu Long',
    'An Giang': 'Đồng bằng sông Cửu Long',
    'Kiên Giang': 'Đồng bằng sông Cửu Long',
    'Cần Thơ': 'Đồng bằng sông Cửu Long',
    'Hậu Giang': 'Đồng bằng sông Cửu Long',
    'Sóc Trăng': 'Đồng bằng sông Cửu Long',
    'Bạc Liêu': 'Đồng bằng sông Cửu Long',
    'Cà Mau': 'Đồng bằng sông Cửu Long'
}


def parse_number(value):
    """
    Chuyển đổi giá trị số từ format tiếng Việt (dấu chấm = phân cách hàng nghìn, dấu phẩy = thập phân)
    """
    if pd.isna(value) or value == '..' or value == '..':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    # Xử lý format tiếng Việt: 1.234,56 -> 1234.56
    value = str(value).strip()
    value = value.replace('.', '').replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return np.nan


def load_marriage_data(filepath='data/marriage-data-2019-2024.xlsx'):
    """
    Đọc dữ liệu số cuộc kết hôn theo tỉnh/thành (2019-2024)
    
    Returns:
    --------
    pd.DataFrame với columns: ['tinh_thanh', 'vung_kinh_te', 'vung_mien', 2019, 2020, 2021, 2022, 2023, 2024]
    """
    df = pd.read_excel(filepath, sheet_name='Sheet3', header=None)
    
    # Xác định năm từ header
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    # Lấy dữ liệu từ row 3 trở đi (bỏ header)
    data_rows = []
    for idx in range(3, len(df)):
        row = df.iloc[idx]
        tinh_thanh = str(row[0]).strip() if pd.notna(row[0]) else None
        
        if tinh_thanh and tinh_thanh not in ['CẢ NƯỚC', 'NaN', ''] and tinh_thanh not in VUNG_MIEN.keys():
            values = [parse_number(row[i]) for i in range(1, 7)]
            data_rows.append([tinh_thanh] + values)
    
    # Tạo DataFrame
    columns = ['tinh_thanh'] + years
    df_result = pd.DataFrame(data_rows, columns=columns)
    
    # Thêm vùng kinh tế và vùng miền
    df_result['vung_kinh_te'] = df_result['tinh_thanh'].map(TINH_THEO_VUNG)
    df_result['vung_mien'] = df_result['vung_kinh_te'].map(VUNG_MIEN)
    
    # Sắp xếp lại columns
    df_result = df_result[['tinh_thanh', 'vung_kinh_te', 'vung_mien'] + years]
    
    return df_result


def load_tfr_data(filepath='data/birth-VN.xlsx'):
    """
    Đọc dữ liệu Tổng tỷ suất sinh (TFR) theo tỉnh/thành (2019-2024)
    
    Returns:
    --------
    pd.DataFrame với columns: ['tinh_thanh', 'vung_kinh_te', 'vung_mien', 2019, 2020, 2021, 2022, 2023, 2024]
    """
    df = pd.read_excel(filepath, sheet_name='Sheet6', header=None)
    
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    data_rows = []
    for idx in range(2, len(df)):
        row = df.iloc[idx]
        tinh_thanh = str(row[0]).strip() if pd.notna(row[0]) else None
        
        if tinh_thanh and tinh_thanh not in ['CẢ NƯỚC', 'NaN', '', 'Hà Tây'] and tinh_thanh not in VUNG_MIEN.keys():
            values = [parse_number(row[i]) for i in range(1, 7)]
            data_rows.append([tinh_thanh] + values)
    
    columns = ['tinh_thanh'] + years
    df_result = pd.DataFrame(data_rows, columns=columns)
    
    df_result['vung_kinh_te'] = df_result['tinh_thanh'].map(TINH_THEO_VUNG)
    df_result['vung_mien'] = df_result['vung_kinh_te'].map(VUNG_MIEN)
    
    df_result = df_result[['tinh_thanh', 'vung_kinh_te', 'vung_mien'] + years]
    
    return df_result


def load_population_data(filepath='data/population-VN.xlsx'):
    """
    Đọc dữ liệu dân số theo tỉnh/thành (2019)
    
    Returns:
    --------
    pd.DataFrame với columns: ['tinh_thanh', 'dien_tich', 'dan_so_2019', 'mat_do']
    """
    df = pd.read_excel(filepath, sheet_name='Sheet1', header=None)
    
    data_rows = []
    for idx in range(3, len(df)):
        row = df.iloc[idx]
        tinh_thanh = str(row[0]).strip() if pd.notna(row[0]) else None
        
        if tinh_thanh and tinh_thanh not in ['CẢ NƯỚC', 'NaN', ''] and tinh_thanh not in VUNG_MIEN.keys():
            dien_tich = parse_number(row[1])
            dan_so = parse_number(row[2])
            mat_do = parse_number(row[12]) if len(row) > 12 else None
            data_rows.append([tinh_thanh, dien_tich, dan_so, mat_do])
    
    df_result = pd.DataFrame(data_rows, columns=['tinh_thanh', 'dien_tich', 'dan_so_2019', 'mat_do'])
    
    df_result['vung_kinh_te'] = df_result['tinh_thanh'].map(TINH_THEO_VUNG)
    df_result['vung_mien'] = df_result['vung_kinh_te'].map(VUNG_MIEN)
    
    return df_result


def create_combined_dataset(data_dir='data'):
    """
    Tạo dataset tổng hợp từ các nguồn dữ liệu
    
    Returns:
    --------
    pd.DataFrame: Dataset tổng hợp với các features:
        - tinh_thanh: Tên tỉnh/thành
        - vung_kinh_te: Vùng kinh tế (6 vùng)
        - vung_mien: Miền (Bắc/Trung/Nam)
        - nam: Năm (2019-2024)
        - so_ket_hon: Số cuộc kết hôn
        - tfr: Tổng tỷ suất sinh
        - dan_so: Dân số (năm 2019)
        - ty_le_ket_hon: Tỷ lệ kết hôn trên 1000 dân
    """
    # Load data
    df_marriage = load_marriage_data(os.path.join(data_dir, 'marriage-data-2019-2024.xlsx'))
    df_tfr = load_tfr_data(os.path.join(data_dir, 'birth-VN.xlsx'))
    df_pop = load_population_data(os.path.join(data_dir, 'population-VN.xlsx'))
    
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    # Melt marriage data
    df_marriage_long = df_marriage.melt(
        id_vars=['tinh_thanh', 'vung_kinh_te', 'vung_mien'],
        value_vars=years,
        var_name='nam',
        value_name='so_ket_hon'
    )
    
    # Melt TFR data
    df_tfr_long = df_tfr.melt(
        id_vars=['tinh_thanh', 'vung_kinh_te', 'vung_mien'],
        value_vars=years,
        var_name='nam',
        value_name='tfr'
    )
    
    # Merge datasets
    df_combined = df_marriage_long.merge(
        df_tfr_long[['tinh_thanh', 'nam', 'tfr']], 
        on=['tinh_thanh', 'nam'], 
        how='left'
    )
    
    df_combined = df_combined.merge(
        df_pop[['tinh_thanh', 'dan_so_2019', 'mat_do']], 
        on='tinh_thanh', 
        how='left'
    )
    
    # Tính tỷ lệ kết hôn trên 1000 dân
    df_combined['ty_le_ket_hon'] = (df_combined['so_ket_hon'] / df_combined['dan_so_2019']) * 1000
    
    # Tạo features cho ML
    df_combined = create_ml_features(df_combined)
    
    return df_combined


def create_ml_features(df):
    """
    Tạo các features cho Machine Learning
    """
    df = df.copy()
    
    # Phân loại TFR
    df['tfr_level'] = pd.cut(
        df['tfr'], 
        bins=[0, 1.8, 2.1, 3.0],
        labels=['Thấp', 'Trung bình', 'Cao'],
        include_lowest=True
    )
    
    # Phân loại tỷ lệ kết hôn
    q33, q66 = df['ty_le_ket_hon'].quantile([0.33, 0.66])
    df['marriage_rate_level'] = pd.cut(
        df['ty_le_ket_hon'],
        bins=[0, q33, q66, df['ty_le_ket_hon'].max()],
        labels=['Thấp', 'Trung bình', 'Cao'],
        include_lowest=True
    )
    
    # Phân loại mật độ dân số
    df['urban_rural'] = df['mat_do'].apply(
        lambda x: 'Đô thị' if x > 1000 else ('Bán đô thị' if x > 500 else 'Nông thôn')
    )
    
    # Target: xu hướng kết hôn (tăng/giảm so với năm trước)
    df = df.sort_values(['tinh_thanh', 'nam'])
    df['so_ket_hon_prev'] = df.groupby('tinh_thanh')['so_ket_hon'].shift(1)
    df['trend'] = (df['so_ket_hon'] >= df['so_ket_hon_prev']).astype(int)
    df['trend_label'] = df['trend'].map({0: 'Giảm', 1: 'Tăng/Giữ nguyên'})
    
    return df


def get_summary_by_region(df):
    """
    Tổng hợp dữ liệu theo vùng miền
    """
    summary = df.groupby(['vung_mien', 'nam']).agg({
        'so_ket_hon': 'sum',
        'tfr': 'mean',
        'dan_so_2019': 'sum',
        'ty_le_ket_hon': 'mean'
    }).reset_index()
    
    return summary


def get_summary_by_year(df):
    """
    Tổng hợp dữ liệu theo năm
    """
    summary = df.groupby('nam').agg({
        'so_ket_hon': 'sum',
        'tfr': 'mean',
        'ty_le_ket_hon': 'mean'
    }).reset_index()
    
    return summary


if __name__ == "__main__":
    # Test loading data
    print("="*60)
    print("TESTING DATA LOADER")
    print("="*60)
    
    # Test từng hàm
    print("\n1. Loading Marriage Data...")
    df_marriage = load_marriage_data()
    print(f"   Shape: {df_marriage.shape}")
    print(df_marriage.head())
    
    print("\n2. Loading TFR Data...")
    df_tfr = load_tfr_data()
    print(f"   Shape: {df_tfr.shape}")
    print(df_tfr.head())
    
    print("\n3. Loading Population Data...")
    df_pop = load_population_data()
    print(f"   Shape: {df_pop.shape}")
    print(df_pop.head())
    
    print("\n4. Creating Combined Dataset...")
    df_combined = create_combined_dataset()
    print(f"   Shape: {df_combined.shape}")
    print(f"   Columns: {df_combined.columns.tolist()}")
    print(df_combined.head(10))
    
    print("\n5. Summary by Region...")
    summary_region = get_summary_by_region(df_combined)
    print(summary_region)
    
    print("\n6. Summary by Year...")
    summary_year = get_summary_by_year(df_combined)
    print(summary_year)
    
    # Lưu dataset
    df_combined.to_csv('data/combined_data.csv', index=False, encoding='utf-8-sig')
    print("\n✓ Đã lưu dataset vào: data/combined_data.csv")
