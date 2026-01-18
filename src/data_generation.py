# =============================================================================
# ĐỒ ÁN: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, 
#        sinh con của giới trẻ Việt Nam (18–35)
# =============================================================================
# File: data_generation.py
# Mô tả: Sinh dữ liệu synthetic microdata (≥3000 cá nhân) tham số hóa theo nso
# =============================================================================

import numpy as np
import pandas as pd
import os

def generate_synthetic_data(n_samples=3000, random_state=2026):
    """
    Sinh dữ liệu synthetic cho phân tích xu hướng kết hôn, sinh con
    
    Parameters:
    -----------
    n_samples : int
        Số lượng mẫu cần sinh (mặc định 3000)
    random_state : int
        Seed để tái tạo kết quả
        
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa dữ liệu synthetic
    """
    np.random.seed(random_state)
    
    # 1) Khung dữ liệu danh mục
    age = np.random.randint(18, 36, n_samples)
    sex = np.random.choice(["Nam", "Nữ"], n_samples, p=[0.49, 0.51])
    region = np.random.choice(["Bắc", "Trung", "Nam"], n_samples, p=[0.45, 0.20, 0.35])
    urban_rural = np.random.choice(["Đô thị", "Nông thôn"], n_samples, p=[0.44, 0.56])
    education = np.random.choice(["≤THPT", "CĐ", "ĐH", ">ĐH"], n_samples, p=[0.48, 0.20, 0.28, 0.04])
    
    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "region": region,
        "urban_rural": urban_rural,
        "education": education
    })
    
    # 2) Thu nhập lognormal (đơn vị: nghìn VNĐ/tháng)
    base = np.random.lognormal(mean=8.05, sigma=0.55, size=n_samples)
    edu_mult = df["education"].map({"≤THPT": 0.85, "CĐ": 1.0, "ĐH": 1.18, ">ĐH": 1.35})
    urb_mult = np.where(df["urban_rural"] == "Đô thị", 1.10, 0.95)
    df["income_pc"] = base * edu_mult * urb_mult
    
    # 3) Tạo age_group & income_level
    df["age_group"] = pd.cut(
        df["age"], 
        bins=[17, 24, 29, 35],
        labels=["18-24", "25-29", "30-35"], 
        include_lowest=True
    )
    
    q33, q66 = df["income_pc"].quantile([0.33, 0.66])
    df["income_level"] = df["income_pc"].apply(
        lambda x: "low" if x < q33 else ("mid" if x < q66 else "high")
    )
    
    # 4) Xác suất logistic tham số hóa theo nso
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def logit_married(r):
        """Tính logit cho xác suất kết hôn"""
        base = -2.2 + {"18-24": -0.6, "25-29": 0.6, "30-35": 1.0}[r.age_group]
        base += -0.25 if r.urban_rural == "Đô thị" else 0.25
        base += {"≤THPT": 0.25, "CĐ": 0.05, "ĐH": -0.20, ">ĐH": -0.35}[r.education]
        base += -0.20 if r.region == "Nam" else (0.05 if r.region == "Bắc" else 0.0)
        return base
    
    def logit_child(r):
        """Tính logit cho xác suất có con"""
        base = -2.8 + {"18-24": -0.7, "25-29": 0.6, "30-35": 1.2}[r.age_group]
        base += -0.35 if r.urban_rural == "Đô thị" else 0.35
        base += {"≤THPT": 0.30, "CĐ": 0.05, "ĐH": -0.25, ">ĐH": -0.40}[r.education]
        base += -0.15 if r.region == "Nam" else (0.05 if r.region == "Bắc" else 0.0)
        return base
    
    # Tính xác suất và sinh biến target
    p_married = df.apply(logit_married, axis=1).pipe(sigmoid)
    p_child = df.apply(logit_child, axis=1).pipe(sigmoid)
    df["marital_class"] = (np.random.rand(n_samples) < p_married).astype(int)
    df["has_child"] = (np.random.rand(n_samples) < p_child).astype(int)
    
    # 5) Sắp xếp lại cột kết quả
    df = df[["age", "age_group", "sex", "region", "urban_rural", "education",
             "income_pc", "income_level", "has_child", "marital_class"]]
    
    return df


def describe_data(df):
    """In thống kê mô tả dữ liệu"""
    print("=" * 60)
    print("THỐNG KÊ MÔ TẢ DỮ LIỆU")
    print("=" * 60)
    print(f"\nSố lượng mẫu: {len(df)}")
    print(f"\nPhân phối biến target (marital_class):")
    print(df["marital_class"].value_counts())
    print(f"\nTỷ lệ kết hôn: {df['marital_class'].mean():.2%}")
    
    print(f"\nPhân phối theo nhóm tuổi:")
    print(df["age_group"].value_counts())
    
    print(f"\nPhân phối theo giới tính:")
    print(df["sex"].value_counts())
    
    print(f"\nPhân phối theo vùng:")
    print(df["region"].value_counts())
    
    print(f"\nPhân phối theo khu vực:")
    print(df["urban_rural"].value_counts())
    
    print(f"\nPhân phối theo học vấn:")
    print(df["education"].value_counts())
    
    print(f"\nThống kê thu nhập (income_pc):")
    print(df["income_pc"].describe())


if __name__ == "__main__":
    # Sinh dữ liệu
    print("Đang sinh dữ liệu synthetic...")
    df = generate_synthetic_data(n_samples=3000, random_state=2026)
    
    # Hiển thị thống kê
    describe_data(df)
    
    # Lưu file CSV
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "marriage_data.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Da luu du lieu vao: {output_path}")
    print(f"[OK] Kich thuoc: {df.shape}")
