# Dự báo Tình trạng Hôn nhân Giới trẻ Việt Nam

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Mô tả

Đồ án áp dụng các thuật toán **Machine Learning** (Decision Tree, Naive Bayes) để phân tích và dự báo **tình trạng hôn nhân (MARST)** của giới trẻ Việt Nam (18-35 tuổi) dựa trên dữ liệu điều tra dân số **IPUMS International** (Vietnam Census 2009, 2019).

### Biến mục tiêu (Target Variable)

| Biến | Tên gốc | Giá trị | Mô tả |
|------|---------|---------|-------|
| **Y_married** | MARST | `0` | Chưa kết hôn (Single/Never married) |
| | | `1` | Đã kết hôn (Married/In union) |

> **Lưu ý**: Biến mục tiêu duy nhất là **tình trạng hôn nhân (MARST)**, không sử dụng tuổi làm biến mục tiêu.

## Tính năng chính

- **Phân tích dữ liệu**: Trực quan hóa và khám phá dữ liệu điều tra dân số
- **Huấn luyện mô hình**: So sánh nhiều thuật toán ML (Decision Tree, Naive Bayes)
- **Dự báo tình trạng hôn nhân**: Dự đoán khả năng kết hôn dựa trên đặc điểm nhân khẩu học
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
| `living_area_level` | Diện tích nhà ở | Nhỏ, Trung bình, Khá, Rộng |
| `household_size_group` | Quy mô hộ gia đình | 1-2, 3-4, 5-6, >6 người |

## Nguồn dữ liệu

| File | Mô tả | Nguồn |
|------|-------|-------|
| `ipumsi_data.csv` | Dữ liệu điều tra dân số gốc | [IPUMS International](https://international.ipums.org/) |
| `ipums_processed.csv` | Dữ liệu đã tiền xử lý | Tự tạo |
| `panel_microdata.csv` | Dữ liệu panel theo thời gian | Tự tạo |

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
│   ├── ipumsi_data.csv          # Dữ liệu IPUMS (không upload GitHub)
│   ├── ipums_processed.csv      # Dữ liệu đã xử lý
│   ├── panel_microdata.csv      # Dữ liệu panel
│   └── clean-data.docx          # Tài liệu mô tả dữ liệu
│
├── models/                   # Mô hình đã huấn luyện
│   ├── decision_tree_entropy*.pkl
│   ├── decision_tree_gini*.pkl
│   ├── naive_bayes_*.pkl
│   └── pipe_*.pkl
│
├── outputs/                  # Kết quả đầu ra
│   ├── figures/              # Biểu đồ
│   ├── model_comparison*.csv    # So sánh mô hình
│   └── feature_importance*.csv  # Độ quan trọng features
│
├── src/                      # Mã nguồn
│   ├── ipums_data_processor.py  # Xử lý dữ liệu IPUMS
│   ├── decision_tree_model.py   # Mô hình Decision Tree
│   ├── naive_bayes_model.py     # Mô hình Naive Bayes
│   ├── visualization.py         # Visualization
│   └── ...
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
| **Tổng quan dữ liệu** | Xem thống kê và phân bố dữ liệu |
| **Khám phá dữ liệu** | Phân tích chi tiết các biến |
| **Huấn luyện mô hình** | Train và đánh giá các mô hình ML |
| **So sánh mô hình** | So sánh hiệu suất các thuật toán |
| **Dự báo** | Dự đoán tình trạng hôn nhân |
| **AI Assistant** | Chat với Gemini AI để phân tích |

### Chạy Pipeline từ command line

```bash
# Xử lý dữ liệu IPUMS
python src/ipums_data_processor.py

# Huấn luyện mô hình
python src/train_marriage_model.py
```

## Kết quả

### Hiệu suất mô hình (trên dữ liệu IPUMS)

| Mô hình | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|:--------:|:---------:|:------:|:--------:|:-------:|
| **Decision Tree (Entropy)** | 0.85 | 0.83 | 0.87 | 0.85 | 0.91 |
| **Decision Tree (Gini)** | 0.85 | 0.83 | 0.87 | 0.85 | 0.91 |
| **Naive Bayes (Gaussian)** | 0.78 | 0.76 | 0.81 | 0.78 | 0.85 |

### Các biến quan trọng

1. **AGE** - Tuổi: Yếu tố quan trọng nhất
2. **EDATTAIN** - Trình độ học vấn: Ảnh hưởng đáng kể
3. **URBAN** - Thành thị/Nông thôn: Phân biệt rõ ràng
4. **SEX** - Giới tính: Có sự khác biệt giữa nam và nữ
5. **REGION** - Vùng miền: Khác biệt văn hóa vùng miền

## Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **Python 3.9+** | Ngôn ngữ lập trình chính |
| **Streamlit** | Web framework |
| **Scikit-learn** | Machine Learning algorithms |
| **Pandas & NumPy** | Xử lý dữ liệu |
| **Plotly & Matplotlib** | Visualization |
| **Google Generative AI** | Gemini AI integration |

## Nhóm thực hiện

- **Môn học**: Khai thác dữ liệu và truyền thông xã hội
- **Trường**: [Tên trường]
- **Giảng viên hướng dẫn**: [Tên giảng viên]

## Links

| Tài nguyên | URL |
|------------|-----|
| Repository | https://github.com/DungNhiedu/data-mining |
| IPUMS International | https://international.ipums.org/ |
| Streamlit Docs | https://docs.streamlit.io/ |
| Scikit-learn Docs | https://scikit-learn.org/stable/ |

## License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

**Nếu project hữu ích, hãy cho một star!**
