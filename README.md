# Du bao Tinh trang Hon nhan Gioi tre Viet Nam

## Mo ta

Do an ap dung cac thuat toan **Machine Learning** (Decision Tree, Naive Bayes, Random Forest, Logistic Regression) de phan tich va du bao **tinh trang hon nhan (MARST)** cua gioi tre Viet Nam (18-35 tuoi) dua tren du lieu dieu tra dan so **IPUMS International** (Vietnam Census 2009, 2019).

### Bien muc tieu (Target Variable)

| Bien | Ten goc | Gia tri | Mo ta |
|------|---------|---------|-------|
| **Y_married** | MARST | `0` | Chua ket hon (Single/Never married) |
| | | `1` | Da ket hon (Married/In union) |

> **Luu y**: Bien muc tieu duy nhat la **tinh trang hon nhan (MARST)**, khong su dung tuoi lam bien muc tieu.

## Tinh nang chinh

- **Phan tich du lieu**: Truc quan hoa va kham pha du lieu dieu tra dan so
- **Huan luyen mo hinh**: So sanh 5 thuat toan ML (Decision Tree Entropy, Decision Tree Gini, Naive Bayes, Random Forest, Logistic Regression)
- **Du bao tinh trang hon nhan**: Du doan kha nang ket hon dua tren dac diem nhan khau hoc
- **Luat IF-THEN**: Trich xuat luat tu cay quyet dinh de giai thich ket qua
- **AI Assistant**: Tich hop Gemini AI de phan tich va dua ra khuyen nghi

## Cac bien dau vao (Features)

| Bien | Mo ta | Gia tri |
|------|-------|---------|
| `year` | Nam dieu tra | 2009, 2019 |
| `age` | Tuoi | 18-35 |
| `age_group` | Nhom tuoi | 18-24, 25-29, 30-35 |
| `sex` | Gioi tinh | Nam, Nu |
| `education_level` | Trinh do hoc van | <=THPT, DH/CD+ |
| `urban_rural` | Khu vuc sinh song | Do thi, Nong thon |
| `region` | Vung mien | Bac, Trung, Nam |
| `home_ownership` | So huu nha | 0 (Khong), 1 (Co) |
| `living_area` | Dien tich nha o | m2 |
| `living_area_level` | Muc dien tich nha o | Nho, Trung binh, Kha, Rong |
| `household_size` | Quy mo ho gia dinh | So nguoi |
| `household_size_group` | Nhom quy mo ho | 1-2, 3-4, 5-6, >6 nguoi |

## Nguon du lieu

| File | Mo ta | Nguon |
|------|-------|-------|
| `ipumsi_data.csv` | Du lieu dieu tra dan so goc | [IPUMS International](https://international.ipums.org/) |
| `ipums_processed.csv` | Du lieu da tien xu ly (6,128,957 ban ghi) | Tu tao |
| `panel_microdata.csv` | Du lieu panel theo thoi gian | Tu tao |

> **Tai du lieu**: Cac file du lieu lon can tai tu [IPUMS International](https://international.ipums.org/).

## Cau truc thu muc

```
data-mining/
|-- app_panel.py              # Ung dung Streamlit chinh
|-- requirements.txt          # Dependencies
|-- README.md                 # Huong dan nay
|-- .env.example              # Mau cau hinh API key
|
|-- data/                     # Du lieu
|   |-- ipumsi_data.csv          # Du lieu IPUMS (khong upload GitHub)
|   |-- ipums_processed.csv      # Du lieu da xu ly
|   |-- panel_microdata.csv      # Du lieu panel
|   |-- clean-data.docx          # Tai lieu mo ta du lieu
|   +-- CTDA.docx                # Chu thich du an
|
|-- models/                   # Mo hinh da huan luyen
|   |-- decision_tree_entropy_ipums.pkl
|   |-- decision_tree_gini_ipums.pkl
|   |-- naive_bayes_ipums.pkl
|   |-- random_forest_ipums.pkl
|   |-- logistic_regression_ipums.pkl
|   +-- feature_names_ipums.pkl
|
|-- outputs/                  # Ket qua dau ra
|   |-- figures/                 # Bieu do
|   |-- model_comparison_ipums.csv
|   |-- feature_importance_decision_tree_entropy.csv
|   |-- feature_importance_decision_tree_gini.csv
|   |-- feature_importance_random_forest.csv
|   +-- feature_importance_logistic_regression.csv
|
|-- src/                      # Ma nguon
|   |-- ipums_data_processor.py  # Xu ly du lieu IPUMS
|   |-- train_marriage_model.py  # Huan luyen cac mo hinh
|   |-- decision_tree_model.py   # Mo hinh Decision Tree
|   |-- naive_bayes_model.py     # Mo hinh Naive Bayes
|   |-- visualization.py         # Visualization
|   +-- ...
|
|-- notebooks/                # Jupyter notebooks
|   |-- analysis.ipynb           # Phan tich du lieu
|   +-- model_comparison.ipynb   # So sanh mo hinh
|
+-- docs/                     # Tai lieu
    +-- VARIABLES_DESIGN.md      # Thiet ke bien
```

## Cai dat

### 1. Clone repository

```bash
git clone https://github.com/DungNhiedu/data-mining.git
cd data-mining
```

### 2. Tao moi truong ao

```bash
# Tao virtual environment
python -m venv .venv

# Kich hoat (macOS/Linux)
source .venv/bin/activate

# Kich hoat (Windows)
.venv\Scripts\activate
```

### 3. Cai dat dependencies

```bash
pip install -r requirements.txt
```

### 4. Cau hinh API Key (tuy chon)

De su dung tinh nang AI Assistant voi Gemini:

```bash
# Tao file .env tu mau
cp .env.example .env

# Chinh sua file .env va them API key
GEMINI_API_KEY=your_gemini_api_key_here
```

Lay API key tai: https://makersuite.google.com/app/apikey

## Su dung

### Chay ung dung Streamlit

```bash
streamlit run app_panel.py
```

Ung dung se mo tai: http://localhost:8501

### Cac tab trong ung dung

| Tab | Chuc nang |
|-----|-----------|
| **Du bao ca nhan** | Du doan tinh trang hon nhan tu thong tin ca nhan |
| **Phan tich du lieu** | Xem thong ke va phan bo du lieu IPUMS |
| **Luat IF-THEN** | Xem cac luat duoc trich xuat tu Decision Tree |
| **So sanh mo hinh** | So sanh hieu suat 5 thuat toan ML |
| **Yeu to anh huong** | Phan tich Feature Importance tu cac mo hinh |
| **Du bao boi AI** | Chat voi Gemini AI de phan tich xu huong |

### Chay Pipeline tu command line

```bash
# Xu ly du lieu IPUMS
python src/ipums_data_processor.py

# Huan luyen mo hinh
python src/train_marriage_model.py
```

## Ket qua

### Hieu suat mo hinh (tren du lieu IPUMS - 6,128,957 ban ghi)

| Mo hinh | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|:--------:|:---------:|:------:|:--------:|:-------:|
| **Decision Tree (Entropy)** | 82.40% | 84.05% | 90.52% | 87.17% | 88.87% |
| **Decision Tree (Gini)** | 82.43% | 84.10% | 90.48% | 87.18% | 88.89% |
| **Naive Bayes (Gaussian)** | 77.21% | 84.14% | 80.68% | 82.37% | 81.61% |
| **Random Forest** | 82.17% | 83.91% | 90.31% | 86.99% | 88.71% |
| **Logistic Regression** | 81.27% | 83.95% | 88.55% | 86.19% | 86.90% |

> **Ghi chu**: Naive Bayes su dung StandardScaler truoc GaussianNB de calibrate xac suat tot hon.

### Cac bien quan trong (Feature Importance)

Dua tren phan tich tu ca 5 mo hinh:

1. **Tuoi (age)** - Yeu to quan trong nhat (54.4% DT-Entropy, 31% RF)
2. **Quy mo ho gia dinh (household_size)** - Top 2 (18-25% o tat ca mo hinh)
3. **Nhom tuoi 18-24** - Nhom co ty le ket hon thap nhat (46.9% DT-Gini)
4. **Gioi tinh** - Nu ket hon 68% > Nam 64%
5. **Khu vuc (Do thi/Nong thon)** - Do thi 69% > Nong thon 63%
6. **Trinh do hoc van** - DH/CD+ 67% > <=THPT 65%
7. **Vung mien** - Bac 68% > Nam 66% > Trung 63%
8. **So huu nha** - Co nha 67% vs Khong co 65%

## Cong nghe su dung

| Cong nghe | Muc dich |
|-----------|----------|
| **Python 3.9+** | Ngon ngu lap trinh chinh |
| **Streamlit** | Web framework |
| **Scikit-learn** | Machine Learning algorithms |
| **Pandas va NumPy** | Xu ly du lieu |
| **Plotly va Matplotlib** | Visualization |
| **Google Generative AI** | Gemini AI integration |

## Nhom thuc hien

- **Mon hoc**: Khai thac du lieu va truyen thong xa hoi
- **Truong**: [Ten truong]
- **Giang vien huong dan**: [Ten giang vien]

## Links

| Tai nguyen | URL |
|------------|-----|
| Repository | https://github.com/DungNhiedu/data-mining |
| IPUMS International | https://international.ipums.org/ |
| Streamlit Docs | https://docs.streamlit.io/ |
| Scikit-learn Docs | https://scikit-learn.org/stable/ |

## License

MIT License - Xem file [LICENSE](LICENSE) de biet them chi tiet.

---

**Neu project huu ich, hay cho mot star!**
