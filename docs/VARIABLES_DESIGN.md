# THIET KE BIEN DU BAO XAC SUAT KET HON

## Nguon du lieu: IPUMS International - Vietnam Census (2009 & 2019)

Du lieu duoc lay tu IPUMS International (Minnesota Population Center), bao gom:
- **Tong dieu tra dan so Viet Nam 2009**
- **Tong dieu tra dan so Viet Nam 2019**

Tong so ban ghi sau xu ly: **~6.1 trieu** nguoi trong do tuoi 18-35

---

## 1. BIEN MUC TIEU (Target Variable)

| Bien | Mo ta | Gia tri | Nguon IPUMS |
|------|-------|---------|-------------|
| `Y_married` | Tinh trang hon nhan | 0 = Chua ket hon, 1 = Da ket hon | MARST |

---

## 2. BIEN DAU VAO (Features)

### 2.1. Thong tin nhan khau hoc

| Bien | Mo ta | Gia tri | Nguon IPUMS |
|------|-------|---------|-------------|
| `age` | Tuoi | 18-35 | AGE |
| `age_group` | Nhom tuoi | "18-24", "25-29", "30-35" | Tinh tu AGE |
| `sex` | Gioi tinh | "Nam", "Nu" | SEX (1=Nam, 2=Nu) |

### 2.2. Trinh do hoc van

| Bien | Mo ta | Gia tri | Nguon IPUMS |
|------|-------|---------|-------------|
| `education` | Trinh do hoc van chi tiet | "<=Tieu hoc", "THCS", "THPT", "DH/CD+" | EDATTAIN |
| `education_level` | Nhom hoc van (cho mo hinh) | "<=THPT", "DH/CD+" | EDATTAIN |

**Ma goc IPUMS:**
- 1 = Less than primary completed -> <=Tieu hoc
- 2 = Primary completed -> THCS
- 3 = Secondary completed -> THPT
- 4 = University completed -> DH/CD+

### 2.3. Dia ly

| Bien | Mo ta | Gia tri | Nguon IPUMS |
|------|-------|---------|-------------|
| `urban_rural` | Thanh thi/Nong thon | "Do thi", "Nong thon" | URBAN |
| `region` | Vung mien | "Bac", "Trung", "Nam" | GEO1_VN |

**Mapping vung mien tu GEO1_VN:**
- **Mien Bac:** Ha Noi, Hai Phong, cac tinh phia Bac (704001-704025)
- **Mien Trung:** Tu Thanh Hoa den Lam Dong (704026-704044)
- **Mien Nam:** Tu Binh Phuoc den Ca Mau (704045-704063)

### 2.4. Dieu kien nha o

| Bien | Mo ta | Gia tri | Nguon IPUMS |
|------|-------|---------|-------------|
| `home_ownership` | So huu nha | 0 = Khong, 1 = Co | OWNERSHIP |
| `living_area` | Dien tich nha (m2) | So thuc | LIVEAREA |
| `living_area_level` | Muc dien tich | "Nho", "Trung binh", "Kha", "Rong" | Quartile cua LIVEAREA |

**Phan nhom dien tich (quartile):**
- Nho: <= Q1 (khoang <=40m2)
- Trung binh: Q1 - Q2 (khoang 40-72m2)
- Kha: Q2 - Q3 (khoang 72-126m2)
- Rong: > Q3 (> 126m2)

### 2.5. Quy mo ho gia dinh

| Bien | Mo ta | Gia tri | Nguon IPUMS |
|------|-------|---------|-------------|
| `household_size` | So nguoi trong ho | So nguyen | Tinh tu SERIAL |
| `household_size_group` | Nhom quy mo ho | "1-2 nguoi", "3-4 nguoi", "5-6 nguoi", ">6 nguoi" | Phan nhom |

### 2.6. Nam dieu tra

| Bien | Mo ta | Gia tri | Nguon IPUMS |
|------|-------|---------|-------------|
| `year` | Nam dieu tra | 2009, 2019 | YEAR |

---

## 3. XU LY DU LIEU

### 3.1. Loc du lieu
- Chi giu nguoi trong do tuoi 18-35
- Loai bo missing values

### 3.2. One-Hot Encoding (cho mo hinh ML)
Cac bien categorical duoc chuyen thanh one-hot:
- `age_group_18-24`, `age_group_25-29`, `age_group_30-35`
- `sex_Nam`, `sex_Nu`
- `education_level_<=THPT`, `education_level_DH/CD+`
- `urban_rural_Do thi`, `urban_rural_Nong thon`
- `region_Bac`, `region_Trung`, `region_Nam`
- `living_area_level_Nho`, `living_area_level_Trung binh`, `living_area_level_Kha`, `living_area_level_Rong`
- `household_size_group_1-2 nguoi`, `household_size_group_3-4 nguoi`, `household_size_group_5-6 nguoi`, `household_size_group_>6 nguoi`

---

## 4. GIA THUYET NGHIEN CUU

Dua tren van hoc va du lieu, cac yeu to anh huong den xac suat ket hon:

### 4.1. Yeu to tich cuc (+)
- **Tuoi cao hon (25-35):** Nguoi truong thanh co xu huong ket hon nhieu hon
- **So huu nha:** Co nha rieng tang kha nang ket hon
- **Do thi:** Co co hoi gap go nhieu hon
- **Hoc van cao:** On dinh kinh te hon
- **Ho gia dinh 3-4 nguoi:** Co su ho tro tu gia dinh

### 4.2. Yeu to tieu cuc (-)
- **Tuoi tre (18-24):** Con hoc tap, chua on dinh
- **Khong co nha:** Rao can kinh te lon
- **Nong thon:** It co hoi viec lam, di cu
- **Dien tich nha nho:** Dieu kien song kho khan

---

## 5. CAU TRUC FILE

```
data/
├── ipumsi_data.csv          # Du lieu IPUMS goc
├── ipums_processed.csv      # Du lieu da xu ly (6.1M records)

models/
├── decision_tree_entropy_ipums.pkl
├── decision_tree_gini_ipums.pkl
├── naive_bayes_ipums.pkl
├── random_forest_ipums.pkl
├── logistic_regression_ipums.pkl
├── feature_names_ipums.pkl  # Danh sach features cho prediction

src/
├── ipums_data_processor.py  # Xu ly du lieu IPUMS
├── train_marriage_model.py  # Huan luyen mo hinh
```

---

## 6. SU DUNG TRONG APP

### Input cho du bao ca nhan:
1. **Tuoi:** Slider 18-35 -> Tu tinh age_group
2. **Gioi tinh:** Selectbox (Nam/Nu)
3. **Vung mien:** Selectbox (Bac/Trung/Nam)
4. **Khu vuc:** Selectbox (Do thi/Nong thon)
5. **Trinh do hoc van:** Selectbox (<=THPT/DH/CD+)
6. **So huu nha:** Selectbox (Co/Khong)
7. **Dien tich nha:** Slider 10-300m2 -> Tu tinh living_area_level
8. **So nguoi trong ho:** Slider 1-12 -> Tu tinh household_size_group

### Output:
- Xac suat ket hon (0-100%)
- Phan loai: Cao (>=50%) / Thap (<50%)
- Cac yeu to anh huong chinh

---

## 7. THONG KE MO TA (Tom tat)

Dua tren du lieu IPUMS Vietnam Census:

| Bien | Ty le ket hon |
|------|---------------|
| **Theo nhom tuoi** | |
| 18-24 | ~35% |
| 25-29 | ~65% |
| 30-35 | ~85% |
| **Theo gioi tinh** | |
| Nam | ~60% |
| Nu | ~68% |
| **Theo so huu nha** | |
| Co nha | ~68% |
| Khong co nha | ~52% |
| **Theo khu vuc** | |
| Do thi | ~69% |
| Nong thon | ~58% |
| **Theo vung mien** | |
| Mien Bac | ~67% |
| Mien Trung | ~60% |
| Mien Nam | ~64% |

---

*Tai lieu nay mo ta chi tiet cac bien duoc thiet ke de du bao xac suat ket hon cua gioi tre Viet Nam (18-35 tuoi) dua tren du lieu Dieu tra Dan so IPUMS.*
