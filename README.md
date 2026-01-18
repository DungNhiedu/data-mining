# Đồ án: Phân tích xu hướng kết hôn, sinh con của giới trẻ Việt Nam

## Mô tả

Đồ án áp dụng các thuật toán **Machine Learning** (Decision Tree, Naive Bayes) để phân tích và dự báo xu hướng kết hôn tại Việt Nam dựa trên dữ liệu **IPUMS International** và dữ liệu panel microdata.

## Tính năng chính

- **Phân tích dữ liệu**: Trực quan hóa và khám phá dữ liệu điều tra dân số
- **Huấn luyện mô hình**: So sánh nhiều thuật toán ML (Decision Tree, Naive Bayes)
- **Dự báo**: Dự đoán xu hướng kết hôn dựa trên các đặc điểm nhân khẩu học
- **AI Assistant**: Tích hợp Gemini AI để phân tích và đưa ra khuyến nghị

## Nguồn dữ liệu

- **ipumsi_data.csv**: Dữ liệu điều tra dân số từ IPUMS International (Vietnam)
- **panel_microdata.csv**: Dữ liệu panel theo thời gian
- **ipums_processed.csv**: Dữ liệu đã tiền xử lý

> **Lưu ý**: Các file dữ liệu lớn, hãy tải từ [IPUMS International](https://international.ipums.org/).

## Cấu trúc thư mục
```
data-mining/
├── app_panel.py                # Ứng dụng Streamlit chính
├── requirements.txt            # Dependencies
├── README.md                   # Hướng dẫn này
├── .env.example                # Mẫu cấu hình API key
│
├── data/                       # Dữ liệu
│   ├── ipumsi_data.csv         # Dữ liệu IPUMS (không có trên GitHub)
│   ├── ipums_processed.csv     # Dữ liệu đã xử lý (không có trên GitHub)
│   ├── panel_microdata.csv     # Dữ liệu panel
│   └── clean-data.docx         # Tài liệu mô tả dữ liệu
│
├── models/                     # Mô hình đã huấn luyện
│   ├── decision_tree_entropy.pkl
│   ├── decision_tree_gini.pkl
│   ├── naive_bayes_gaussian.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── ... (các mô hình khác)
│
├── outputs/                    # Kết quả đầu ra
│   ├── figures/                # Biểu đồ
│   ├── model_comparison.csv    # So sánh mô hình
│   ├── feature_importance_*.csv # Độ quan trọng features
│   └── predictions_2025.csv    # Dự đoán
│
├── src/                        # Mã nguồn
│   ├── __init__.py
│   ├── data_loader.py          # Đọc dữ liệu
│   ├── data_processor.py       # Xử lý dữ liệu
│   ├── preprocessing.py        # Tiền xử lý
│   ├── decision_tree_model.py  # Mô hình Decision Tree
│   ├── naive_bayes_model.py    # Mô hình Naive Bayes
│   ├── models.py               # Các mô hình ML khác
│   ├── visualization.py        # Visualization
│   ├── panel_data_generator.py # Tạo dữ liệu panel
│   └── main.py                 # Pipeline chính
│
├── notebooks/                  # Jupyter notebooks
│   ├── analysis.ipynb          # Phân tích dữ liệu
│   └── model_comparison.ipynb  # So sánh mô hình
│
└── docs/                       # Tài liệu
    └── VARIABLES_DESIGN.md     # Thiết kế biến
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

## 💻 Sử dụng

### Chạy ứng dụng Streamlit

```bash
streamlit run app_panel.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

### Các tab trong ứng dụng

1. **Tổng quan dữ liệu**: Xem thống kê và phân bố dữ liệu
2. **Khám phá dữ liệu**: Phân tích chi tiết các biến
3. **Huấn luyện mô hình**: Train và đánh giá các mô hình ML
4. **So sánh mô hình**: So sánh hiệu suất các thuật toán
5. **Dự báo**: Dự đoán xu hướng kết hôn
6. **AI Assistant**: Chat với Gemini AI để phân tích

### Chạy Pipeline từ command line

```bash
python src/main.py
```

## Kết quả

### Hiệu suất mô hình (trên dữ liệu IPUMS)

| Mô hình | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| Decision Tree (Entropy) | 0.85 | 0.83 | 0.87 | 0.85 | 0.91 |
| Decision Tree (Gini) | 0.85 | 0.83 | 0.87 | 0.85 | 0.91 |
| Naive Bayes (Gaussian) | 0.78 | 0.76 | 0.81 | 0.78 | 0.85 |
| Logistic Regression | 0.82 | 0.80 | 0.84 | 0.82 | 0.89 |
| Random Forest | 0.87 | 0.85 | 0.89 | 0.87 | 0.93 |

### Các biến quan trọng

1. **AGE** (Tuổi): Yếu tố quan trọng nhất
2. **EDATTAIN** (Trình độ học vấn): Ảnh hưởng đáng kể
3. **URBAN** (Thành thị/Nông thôn): Phân biệt rõ ràng
4. **EMPSTAT** (Tình trạng việc làm): Liên quan đến quyết định kết hôn
5. **SEX** (Giới tính): Có sự khác biệt giữa nam và nữ

## Công nghệ sử dụng

- **Python 3.9+**
- **Streamlit**: Web framework
- **Scikit-learn**: Machine Learning
- **Pandas & NumPy**: Xử lý dữ liệu
- **Plotly & Matplotlib**: Visualization
- **Google Generative AI**: Gemini AI integration

## Nhóm thực hiện

- **Môn học**: Khai thác dữ liệu và truyền thông xã hội
- **Giảng viên hướng dẫn**: [Tên giảng viên]

## Links

- **Repository**: https://github.com/DungNhiedu/data-mining
- **IPUMS International**: https://international.ipums.org/
- **Streamlit Documentation**: https://docs.streamlit.io/
