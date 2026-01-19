# =============================================================================
# ĐỒ ÁN: Dự báo Tình trạng Hôn nhân của Giới trẻ (18–35), Giai đoạn 2019–2024
# =============================================================================
# File: panel_data_generator.py
# Mô tả: Tạo panel synthetic microdata với các biến kinh tế vĩ mô
#
# BIẾN MỤC TIÊU (Target Variable):
#   - Y_married: Tình trạng hôn nhân
#     + 0 = Chưa kết hôn
#     + 1 = Đã kết hôn
# =============================================================================

import numpy as np
import pandas as pd
import os

# Cấu hình
REGIONS = ["Bắc", "Trung", "Nam"]
YEARS = list(range(2019, 2025))
np.random.seed(2026)
rng = np.random.default_rng(2026)


def load_macro(path="data/macro_region_year.csv"):
    """Tải hoặc tạo dữ liệu macro vùng-năm"""
    try:
        m = pd.read_csv(path)
        m["region"] = pd.Categorical(m["region"], REGIONS)
        return m
    except:
        # Tạo synthetic macro data
        grid = [(r, y) for r in REGIONS for y in YEARS]
        m = pd.DataFrame(grid, columns=["region", "year"])
        
        # Chỉ số giá nhà (tăng theo năm, khác biệt theo vùng)
        base_hpi = {"Bắc": 1.00, "Trung": 0.97, "Nam": 1.05}
        m["house_price_index"] = [
            100 * base_hpi[r] * (1 + 0.05) ** (y - 2019) 
            for r, y in zip(m.region, m.year)
        ]
        
        # Chỉ số thuê nhà
        m["rental_index"] = [
            100 * base_hpi[r] * (1 + 0.03) ** (y - 2019) 
            for r, y in zip(m.region, m.year)
        ]
        
        # Tỷ lệ thất nghiệp (tăng cao 2020-2021 do COVID)
        m["unemployment_rate"] = [
            2.2 + (1.6 if y in (2020, 2021) else 0.6) + (0.3 if r == "Nam" else 0)
            for r, y in zip(m.region, m.year)
        ]
        
        # CPI
        m["CPI"] = [100 * (1 + 0.03) ** (y - 2019) for y in m.year]
        
        # Chi phí trông trẻ
        m["child_care_cost_index"] = [100 * (1 + 0.04) ** (y - 2019) for y in m.year]
        
        # Ưu đãi thuế gia đình
        m["family_tax_benefit_index"] = [
            100 + (3 if r != "Nam" else 2) * (y - 2019)
            for r, y in zip(m.region, m.year)
        ]
        
        # Lãi suất proxy
        m["interest_rate_proxy"] = [
            6.5 + (1.0 if y == 2023 else 0) - (0.7 if y == 2020 else 0)
            for y in m.year
        ]
        
        return m


def generate_panel_data(N_base=4000, save_path="data/panel_microdata.csv"):
    """Tạo panel data synthetic cho giai đoạn 2019-2024"""
    
    # Tạo base population
    base = pd.DataFrame({
        "id": np.arange(N_base),
        "age_2019": rng.integers(18, 36, N_base),
        "sex": rng.choice(["Nam", "Nữ"], N_base, p=[0.49, 0.51]),
        "region": rng.choice(REGIONS, N_base, p=[0.45, 0.20, 0.35]),
        "urban_rural": rng.choice(["Đô thị", "Nông thôn"], N_base, p=[0.44, 0.56]),
        "education": rng.choice(["≤THPT", "CĐ", "ĐH", ">ĐH"], N_base, p=[0.48, 0.20, 0.28, 0.04]),
        "home_ownership_2019": rng.choice([0, 1], N_base, p=[0.7, 0.3]),
        "prior_marital_2019": rng.choice([0, 1], N_base, p=[0.75, 0.25])
    })
    
    # Load macro data
    macro = load_macro()
    
    rows = []
    for y in YEARS:
        tmp = base.copy()
        tmp["year"] = y
        tmp["age"] = tmp["age_2019"] + (y - 2019)
        
        # Lọc tuổi 18-35
        tmp = tmp[(tmp.age >= 18) & (tmp.age <= 35)].copy()
        
        # Tạo age_group
        tmp["age_group"] = pd.cut(
            tmp["age"], 
            bins=[17, 24, 29, 35],
            labels=["18-24", "25-29", "30-35"]
        )
        
        # Employment status
        tmp["employment_status"] = rng.choice(
            ["FT", "PT", "Unemp"], 
            len(tmp), 
            p=[0.70, 0.20, 0.10]
        )
        
        # Home ownership (có thể tăng theo năm)
        tmp["home_ownership"] = tmp["home_ownership_2019"]
        new_owners = (tmp["age"] >= 28) & (tmp["home_ownership"] == 0) & (rng.random(len(tmp)) < 0.05)
        tmp.loc[new_owners, "home_ownership"] = 1
        
        # Thu nhập hàng tháng (lognormal, phụ thuộc học vấn và đô thị)
        base_income = rng.lognormal(mean=8.5, sigma=0.5, size=len(tmp))
        edu_mult = tmp["education"].map({"≤THPT": 0.8, "CĐ": 1.0, "ĐH": 1.3, ">ĐH": 1.6})
        urb_mult = np.where(tmp["urban_rural"] == "Đô thị", 1.15, 0.90)
        emp_mult = tmp["employment_status"].map({"FT": 1.0, "PT": 0.6, "Unemp": 0.2})
        year_growth = 1 + 0.05 * (y - 2019)
        tmp["income_monthly"] = base_income * edu_mult * urb_mult * emp_mult * year_growth
        
        # Chi tiêu hàng tháng
        tmp["expenditure_monthly"] = tmp["income_monthly"] * rng.uniform(0.5, 0.9, len(tmp))
        
        # Housing cost share
        tmp["housing_cost_share"] = np.where(
            tmp["urban_rural"] == "Đô thị",
            rng.uniform(0.25, 0.45, len(tmp)),
            rng.uniform(0.10, 0.25, len(tmp))
        )
        
        # Debt
        tmp["debt_outstanding"] = rng.exponential(scale=50000, size=len(tmp)) * (1 - tmp["home_ownership"] * 0.3)
        tmp["debt_service_ratio"] = np.clip(tmp["debt_outstanding"] * 0.01 / (tmp["income_monthly"] + 1), 0, 0.6)
        
        # Merge macro data
        tmp = tmp.merge(macro[macro.year == y], on=["region", "year"], how="left")
        
        # Affordability index
        tmp["affordability_index"] = (
            tmp["income_monthly"] * 12 
            / (tmp["house_price_index"] * 1000 + tmp["rental_index"] * 120)
        )
        
        rows.append(tmp)
    
    panel = pd.concat(rows, ignore_index=True).sort_values(["id", "year"])
    
    # Tính income_level và afford_level
    panel["income_level"] = pd.qcut(
        panel["income_monthly"].rank(method="first"), 
        3, 
        labels=["low", "mid", "high"]
    )
    
    panel["afford_level"] = pd.qcut(
        panel["affordability_index"].rank(method="first"), 
        4, 
        labels=["Q1", "Q2", "Q3", "Q4"]
    )
    
    # Tính prior_marital_status và marital_status
    panel = calculate_marital_transitions(panel)
    
    # Lưu file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        panel.to_csv(save_path, index=False)
        print(f"Saved panel data to {save_path}")
    
    return panel


def calculate_marital_transitions(panel):
    """Tính toán chuyển đổi trạng thái hôn nhân"""
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    # Sắp xếp theo id và năm
    panel = panel.sort_values(["id", "year"]).reset_index(drop=True)
    
    # Khởi tạo
    panel["prior_marital_status"] = 0
    panel["marital_status"] = 0
    panel["Y_married"] = 0
    panel["Y_child_next"] = 0
    
    # Xử lý từng cá nhân
    for person_id in panel["id"].unique():
        mask = panel["id"] == person_id
        person_data = panel[mask].copy()
        
        prior_status = person_data["prior_marital_2019"].iloc[0]
        
        for idx in person_data.index:
            row = panel.loc[idx]
            
            # Prior status
            panel.loc[idx, "prior_marital_status"] = prior_status
            
            if prior_status == 0:
                # Tính xác suất kết hôn
                logit = -2.5
                
                # Age effect
                age_effect = {"18-24": -0.8, "25-29": 0.5, "30-35": 1.2}
                logit += age_effect.get(row["age_group"], 0)
                
                # Urban effect
                logit += -0.4 if row["urban_rural"] == "Đô thị" else 0.3
                
                # Education effect
                edu_effect = {"≤THPT": 0.3, "CĐ": 0.1, "ĐH": -0.2, ">ĐH": -0.4}
                logit += edu_effect.get(row["education"], 0)
                
                # Region effect
                region_effect = {"Bắc": 0.1, "Trung": 0, "Nam": -0.2}
                logit += region_effect.get(row["region"], 0)
                
                # Affordability effect
                afford_effect = {"Q1": -0.5, "Q2": -0.2, "Q3": 0.2, "Q4": 0.5}
                logit += afford_effect.get(row["afford_level"], 0)
                
                # Home ownership effect
                logit += 0.4 if row["home_ownership"] == 1 else 0
                
                # Employment effect
                emp_effect = {"FT": 0.2, "PT": -0.1, "Unemp": -0.5}
                logit += emp_effect.get(row["employment_status"], 0)
                
                # Unemployment rate effect (macro)
                logit -= row["unemployment_rate"] * 0.1
                
                # Calculate probability
                prob_marry = sigmoid(logit)
                married_this_year = rng.random() < prob_marry
                
                if married_this_year:
                    panel.loc[idx, "Y_married"] = 1
                    panel.loc[idx, "marital_status"] = 1
                    prior_status = 1
                else:
                    panel.loc[idx, "marital_status"] = 0
            else:
                # Đã kết hôn từ trước
                panel.loc[idx, "marital_status"] = 1
                
                # Tính xác suất sinh con
                logit_child = -3.0
                logit_child += {"18-24": -0.5, "25-29": 0.4, "30-35": 0.8}.get(row["age_group"], 0)
                logit_child += -0.5 if row["urban_rural"] == "Đô thị" else 0.3
                logit_child += {"Q1": -0.4, "Q2": -0.1, "Q3": 0.2, "Q4": 0.4}.get(row["afford_level"], 0)
                logit_child -= row["child_care_cost_index"] * 0.005
                
                prob_child = sigmoid(logit_child)
                panel.loc[idx, "Y_child_next"] = int(rng.random() < prob_child)
    
    return panel


def get_panel_summary(panel):
    """Tạo bảng tổng hợp thống kê"""
    summary = panel.groupby(["year", "region"]).agg({
        "Y_married": ["sum", "mean"],
        "marital_status": "mean",
        "income_monthly": "mean",
        "affordability_index": "mean",
        "unemployment_rate": "first"
    }).round(3)
    
    return summary


if __name__ == "__main__":
    # Tạo panel data
    panel = generate_panel_data(N_base=4000)
    print(f"\nPanel shape: {panel.shape}")
    print(f"\nColumns: {panel.columns.tolist()}")
    print(f"\nY_married distribution:\n{panel['Y_married'].value_counts()}")
    print(f"\nSummary by year:")
    print(panel.groupby("year")["Y_married"].agg(["sum", "mean"]))
