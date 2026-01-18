# THIẾT KẾ BIẾN DỰ BÁO XÁC SUẤT KẾT HÔN

## Nguồn dữ liệu: IPUMS International - Vietnam Census (2009 & 2019)

Dữ liệu được lấy từ IPUMS International (Minnesota Population Center), bao gồm:
- **Tổng điều tra dân số Việt Nam 2009**
- **Tổng điều tra dân số Việt Nam 2019**

Tổng số bản ghi sau xử lý: **~6.1 triệu** người trong độ tuổi 18-35

---

## 1. BIẾN MỤC TIÊU (Target Variable)

| Biến | Mô tả | Giá trị | Nguồn IPUMS |
|------|-------|---------|-------------|
| `Y_married` | Tình trạng hôn nhân | 0 = Chưa kết hôn, 1 = Đã kết hôn | MARST |

---

## 2. BIẾN ĐẦU VÀO (Features)

### 2.1. Thông tin nhân khẩu học

| Biến | Mô tả | Giá trị | Nguồn IPUMS |
|------|-------|---------|-------------|
| `age` | Tuổi | 18-35 | AGE |
| `age_group` | Nhóm tuổi | "18-24", "25-29", "30-35" | Tính từ AGE |
| `sex` | Giới tính | "Nam", "Nữ" | SEX (1=Nam, 2=Nữ) |

### 2.2. Trình độ học vấn

| Biến | Mô tả | Giá trị | Nguồn IPUMS |
|------|-------|---------|-------------|
| `education` | Trình độ học vấn chi tiết | "≤Tiểu học", "THCS", "THPT", "ĐH/CĐ+" | EDATTAIN |
| `education_level` | Nhóm học vấn (cho mô hình) | "≤THPT", "ĐH/CĐ+" | EDATTAIN |

**Mã gốc IPUMS:**
- 1 = Less than primary completed → ≤Tiểu học
- 2 = Primary completed → THCS
- 3 = Secondary completed → THPT
- 4 = University completed → ĐH/CĐ+

### 2.3. Địa lý

| Biến | Mô tả | Giá trị | Nguồn IPUMS |
|------|-------|---------|-------------|
| `urban_rural` | Thành thị/Nông thôn | "Đô thị", "Nông thôn" | URBAN |
| `region` | Vùng miền | "Bắc", "Trung", "Nam" | GEO1_VN |

**Mapping vùng miền từ GEO1_VN:**
- **Miền Bắc:** Hà Nội, Hải Phòng, các tỉnh phía Bắc (704001-704025)
- **Miền Trung:** Từ Thanh Hóa đến Lâm Đồng (704026-704044)
- **Miền Nam:** Từ Bình Phước đến Cà Mau (704045-704063)

### 2.4. Điều kiện nhà ở

| Biến | Mô tả | Giá trị | Nguồn IPUMS |
|------|-------|---------|-------------|
| `home_ownership` | Sở hữu nhà | 0 = Không, 1 = Có | OWNERSHIP |
| `living_area` | Diện tích nhà (m²) | Số thực | LIVEAREA |
| `living_area_level` | Mức diện tích | "Nhỏ", "Trung bình", "Khá", "Rộng" | Quartile của LIVEAREA |

**Phân nhóm diện tích (quartile):**
- Nhỏ: ≤ Q1 (khoảng ≤40m²)
- Trung bình: Q1 - Q2 (khoảng 40-72m²)
- Khá: Q2 - Q3 (khoảng 72-126m²)
- Rộng: > Q3 (> 126m²)

### 2.5. Quy mô hộ gia đình

| Biến | Mô tả | Giá trị | Nguồn IPUMS |
|------|-------|---------|-------------|
| `household_size` | Số người trong hộ | Số nguyên | Tính từ SERIAL |
| `household_size_group` | Nhóm quy mô hộ | "1-2 người", "3-4 người", "5-6 người", ">6 người" | Phân nhóm |

### 2.6. Năm điều tra

| Biến | Mô tả | Giá trị | Nguồn IPUMS |
|------|-------|---------|-------------|
| `year` | Năm điều tra | 2009, 2019 | YEAR |

---

## 3. XỬ LÝ DỮ LIỆU

### 3.1. Lọc dữ liệu
- Chỉ giữ người trong độ tuổi 18-35
- Loại bỏ missing values

### 3.2. One-Hot Encoding (cho mô hình ML)
Các biến categorical được chuyển thành one-hot:
- `age_group_18-24`, `age_group_25-29`, `age_group_30-35`
- `sex_Nam`, `sex_Nữ`
- `education_level_≤THPT`, `education_level_ĐH/CĐ+`
- `urban_rural_Đô thị`, `urban_rural_Nông thôn`
- `region_Bắc`, `region_Trung`, `region_Nam`
- `living_area_level_Nhỏ`, `living_area_level_Trung bình`, `living_area_level_Khá`, `living_area_level_Rộng`
- `household_size_group_1-2 người`, `household_size_group_3-4 người`, `household_size_group_5-6 người`, `household_size_group_>6 người`

---

## 4. GIẢ THUYẾT NGHIÊN CỨU

Dựa trên văn học và dữ liệu, các yếu tố ảnh hưởng đến xác suất kết hôn:

### 4.1. Yếu tố tích cực (+)
- **Tuổi cao hơn (25-35):** Người trưởng thành có xu hướng kết hôn nhiều hơn
- **Sở hữu nhà:** Có nhà riêng tăng khả năng kết hôn
- **Đô thị:** Có cơ hội gặp gỡ nhiều hơn
- **Học vấn cao:** Ổn định kinh tế hơn
- **Hộ gia đình 3-4 người:** Có sự hỗ trợ từ gia đình

### 4.2. Yếu tố tiêu cực (-)
- **Tuổi trẻ (18-24):** Còn học tập, chưa ổn định
- **Không có nhà:** Rào cản kinh tế lớn
- **Nông thôn:** Ít cơ hội việc làm, di cư
- **Diện tích nhà nhỏ:** Điều kiện sống khó khăn

---

## 5. CẤU TRÚC FILE

```
data/
├── ipumsi_data.csv          # Dữ liệu IPUMS gốc
├── ipums_processed.csv      # Dữ liệu đã xử lý (6.1M records)

models/
├── decision_tree_entropy_ipums.pkl
├── decision_tree_gini_ipums.pkl
├── naive_bayes_ipums.pkl
├── random_forest_ipums.pkl
├── logistic_regression_ipums.pkl
├── feature_names_ipums.pkl  # Danh sách features cho prediction

src/
├── ipums_data_processor.py  # Xử lý dữ liệu IPUMS
├── train_marriage_model.py  # Huấn luyện mô hình
```

---

## 6. SỬ DỤNG TRONG APP

### Input cho dự báo cá nhân:
1. **Tuổi:** Slider 18-35 → Tự tính age_group
2. **Giới tính:** Selectbox (Nam/Nữ)
3. **Vùng miền:** Selectbox (Bắc/Trung/Nam)
4. **Khu vực:** Selectbox (Đô thị/Nông thôn)
5. **Trình độ học vấn:** Selectbox (≤THPT/ĐH/CĐ+)
6. **Sở hữu nhà:** Selectbox (Có/Không)
7. **Diện tích nhà:** Slider 10-300m² → Tự tính living_area_level
8. **Số người trong hộ:** Slider 1-12 → Tự tính household_size_group

### Output:
- Xác suất kết hôn (0-100%)
- Phân loại: Cao (≥50%) / Thấp (<50%)
- Các yếu tố ảnh hưởng chính

---

## 7. THỐNG KÊ MÔ TẢ (Tóm tắt)

Dựa trên dữ liệu IPUMS Vietnam Census:

| Biến | Tỷ lệ kết hôn |
|------|---------------|
| **Theo nhóm tuổi** | |
| 18-24 | ~35% |
| 25-29 | ~65% |
| 30-35 | ~85% |
| **Theo giới tính** | |
| Nam | ~60% |
| Nữ | ~68% |
| **Theo sở hữu nhà** | |
| Có nhà | ~68% |
| Không có nhà | ~52% |
| **Theo khu vực** | |
| Đô thị | ~69% |
| Nông thôn | ~58% |
| **Theo vùng miền** | |
| Miền Bắc | ~67% |
| Miền Trung | ~60% |
| Miền Nam | ~64% |

---

*Tài liệu này mô tả chi tiết các biến được thiết kế để dự báo xác suất kết hôn của giới trẻ Việt Nam (18-35 tuổi) dựa trên dữ liệu Điều tra Dân số IPUMS.*
