# Dự báo Tình trạng Hôn nhân Giới trẻ Việt Nam

## Mô tả

Đồ án áp dụng các thuật toán **Machine Learning** (Decision Tree, Naive Bayes, Random Forest, Logistic Regression) để phân tích và dự báo **tình trạng hôn nhân (MARST)** của giới trẻ Việt Nam (18-35 tuổi) dựa trên dữ liệu điều tra dân số **IPUMS International** (Vietnam Census 2009, 2019).

### Biến mục tiêu (Target Variable)

| Biến | Tên gốc | Giá trị | Mô tả |
|------|---------|---------|-------|
| **Y_married** | MARST | `0` | Chưa kết hôn (Single/Never married) |
| | | `1` | Đã kết hôn (Married/In union) |

> **Lưu ý**: Biến mục tiêu duy nhất là **tình trạng hôn nhân (MARST)**, không sử dụng tuổi làm biến mục tiêu.

## Tính năng chính

- **Phân tích dữ liệu**: Trực quan hóa và khám phá dữ liệu điều tra dân số
- **Huấn luyện mô hình**: So sánh 5 thuật toán ML (Decision Tree Entropy, Decision Tree Gini, Naive Bayes, Random Forest, Logistic Regression)
- **Dự báo tình trạng hôn nhân**: Dự đoán khả năng kết hôn dựa trên đặc điểm nhân khẩu học
- **Luật IF-THEN**: Trích xuất luật từ cây quyết định để giải thích kết quả
- **AI Assistant**: Tích hợp Gemini AI để phân tích và đưa ra khuyến nghị

## Các biến đầu vào (Features)

| Biến | Mô tả | Giá trị |
|------|-------|---------|
| `year` | Năm điều tra | 2009, 2019 |
| `age` | Tuổi | 18-35 |
| `age_group` | Nhóm tuổi | 18-24, 25-29, 30-35 |
| `sex` | Giới tính | Nam, Nữ |
| `education_level` | Trình độ học vấn | ≤THPT, ĐH/CĐ+ |
| `urban_rural` | Khu vực sinh sống | Đô thị, Nông thôn |
| `region` | Vùng miền | Bắc, Trung, Nam |
| `home_ownership` | Sở hữu nhà | 0 (Không), 1 (Có) |
| `living_area` | Diện tích nhà ở | m² |
| `living_area_level` | Mức diện tích nhà ở | Nhỏ, Trung bình, Khá, Rộng |
| `household_size` | Quy mô hộ gia đình | Số người |
| `household_size_group` | Nhóm quy mô hộ | 1-2, 3-4, 5-6, >6 người |

## Nguồn dữ liệu

| File | Mô tả | Nguồn |
|------|-------|-------|
| `ipumsi_data.csv` | Dữ liệu điều tra dân số gốc | [IPUMS International](https://international.ipums.org/) |
| `ipums_processed.csv` | Dữ liệu đã tiền xử lý (6,128,957 bản ghi) | Tự tạo |

> **Tải dữ liệu**: Các file dữ liệu lớn cần tải từ [IPUMS International](https://international.ipums.org/).

## Cấu trúc thư mục

```
data-mining/
├── app_panel.py              # Ứng dụng Streamlit chính
├── requirements.txt          # Dependencies
├── README.md                 # Hướng dẫn này
├── .env.example              # Mẫu cấu hình API key
│
├── data/                     # Dữ liệu
│   ├── ipumsi_data.csv          # Dữ liệu IPUMS gốc
│   ├── ipums_processed.csv      # Dữ liệu đã xử lý
│   ├── clean-data.docx          # Tài liệu mô tả dữ liệu
│   └── CTDA.docx                # Chú thích đồ án
│
├── models/                   # Mô hình đã huấn luyện (6 files)
│   ├── decision_tree_entropy_ipums.pkl
│   ├── decision_tree_gini_ipums.pkl
│   ├── naive_bayes_ipums.pkl
│   ├── random_forest_ipums.pkl
│   ├── logistic_regression_ipums.pkl
│   └── feature_names_ipums.pkl
│
├── outputs/                  # Kết quả đầu ra
│   ├── figures/                 # Biểu đồ
│   ├── model_comparison_ipums.csv
│   ├── feature_importance_decision_tree_entropy.csv
│   ├── feature_importance_decision_tree_gini.csv
│   ├── feature_importance_random_forest.csv
│   └── feature_importance_logistic_regression.csv
│
├── src/                      # Mã nguồn (9 files)
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Load dữ liệu
│   ├── data_processor.py        # Xử lý dữ liệu
│   ├── preprocessing.py         # Tiền xử lý, encoding
│   ├── ipums_data_processor.py  # Xử lý dữ liệu IPUMS
│   ├── train_marriage_model.py  # Huấn luyện các mô hình
│   ├── decision_tree_model.py   # Mô hình Decision Tree
│   ├── naive_bayes_model.py     # Mô hình Naive Bayes
│   └── visualization.py         # Visualization
│
├── notebooks/                # Jupyter notebooks
│   ├── analysis.ipynb           # Phân tích dữ liệu
│   └── model_comparison.ipynb   # So sánh mô hình
│
└── docs/                     # Tài liệu
    └── VARIABLES_DESIGN.md      # Thiết kế biến
```

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/DungNhiedu/data-mining.git
cd data-mining
```

### 2. Tạo môi trường ảo

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (macOS/Linux)
source .venv/bin/activate

# Kích hoạt (Windows)
.venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình API Key (tùy chọn)

Để sử dụng tính năng AI Assistant với Gemini:

```bash
# Tạo file .env từ mẫu
cp .env.example .env

# Chỉnh sửa file .env và thêm API key
GEMINI_API_KEY=your_gemini_api_key_here
```

Lấy API key tại: https://makersuite.google.com/app/apikey

## Sử dụng

### Chạy ứng dụng Streamlit

```bash
streamlit run app_panel.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

### Các tab trong ứng dụng

| Tab | Chức năng |
|-----|-----------|
| **Dự báo cá nhân** | Dự đoán tình trạng hôn nhân từ thông tin cá nhân |
| **Phân tích dữ liệu** | Xem thống kê và phân bố dữ liệu IPUMS |
| **Luật IF-THEN** | Xem các luật được trích xuất từ Decision Tree |
| **So sánh mô hình** | So sánh hiệu suất 5 thuật toán ML |
| **Yếu tố ảnh hưởng** | Phân tích Feature Importance từ các mô hình |
| **Dự báo bởi AI** | Chat với Gemini AI để phân tích xu hướng |

### Chạy Pipeline từ command line

```bash
# Xử lý dữ liệu IPUMS
python src/ipums_data_processor.py

# Huấn luyện mô hình
python src/train_marriage_model.py
```

## Kết quả

### Hiệu suất mô hình (trên dữ liệu IPUMS - 6,128,957 bản ghi)

| Mô hình | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|:--------:|:---------:|:------:|:--------:|:-------:|
| **Decision Tree (Entropy)** | 82.40% | 84.05% | 90.52% | 87.17% | 88.87% |
| **Decision Tree (Gini)** | 82.43% | 84.10% | 90.48% | 87.18% | 88.89% |
| **Naive Bayes** | 77.21% | 84.14% | 80.68% | 82.37% | 81.61% |
| **Random Forest** | 82.17% | 83.91% | 90.31% | 86.99% | 88.71% |
| **Logistic Regression** | 81.27% | 83.95% | 88.55% | 86.19% | 86.90% |

### Các biến quan trọng (Feature Importance)

Dựa trên phân tích từ các mô hình:

1. **Tuổi (age)** - Yếu tố quan trọng nhất
2. **Quy mô hộ gia đình (household_size)** - Top 2
3. **Nhóm tuổi 18-24** - Nhóm có tỷ lệ kết hôn thấp nhất
4. **Giới tính** - Nữ kết hôn cao hơn Nam
5. **Khu vực (Đô thị/Nông thôn)**
6. **Trình độ học vấn**
7. **Vùng miền**
8. **Sở hữu nhà**

## Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **Python 3.9+** | Ngôn ngữ lập trình chính |
| **Streamlit** | Web framework |
| **Scikit-learn** | Machine Learning algorithms |
| **Pandas, NumPy** | Xử lý dữ liệu |
| **Plotly, Matplotlib** | Visualization |
| **Google Generative AI** | Gemini AI integration |

## Nhóm thực hiện

- **Môn học**: Khai thác dữ liệu và truyền thông xã hội
- **Trường**:
- **Giảng viên hướng dẫn**: 

## Links

| Tài nguyên | URL |
|------------|-----|
| Repository | https://github.com/DungNhiedu/data-mining |
| IPUMS International | https://international.ipums.org/ |
| Streamlit Docs | https://docs.streamlit.io/ |
| Scikit-learn Docs | https://scikit-learn.org/stable/ |
