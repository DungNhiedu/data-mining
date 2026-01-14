# Đồ án: Phân tích xu hướng kết hôn, sinh con của giới trẻ Việt Nam (18–35)

## Mô tả
Đồ án áp dụng **Cây quyết định (Decision Tree)** và **Naive Bayes** để phân tích và dự báo xu hướng kết hôn tại Việt Nam dựa trên dữ liệu thực từ **Tổng cục Thống kê (nso)** giai đoạn 2019-2024.

## Nguồn dữ liệu
- **marriage-data-2019-2024.xlsx**: Số cuộc kết hôn theo tỉnh/thành (2019-2024)
- **birth-VN.xlsx**: Tổng tỷ suất sinh (TFR) theo tỉnh/thành (2019-2024)
- **population-VN.xlsx**: Dân số, mật độ dân số theo tỉnh/thành

## Cấu trúc thư mục

```
data-mining/
├── data/                       # Dữ liệu
│   ├── marriage-data-2019-2024.xlsx  # Dữ liệu kết hôn nso
│   ├── birth-VN.xlsx                 # Dữ liệu tỷ suất sinh nso
│   ├── population-VN.xlsx            # Dữ liệu dân số nso
│   ├── VN-2019.xlsx                  # Dữ liệu bổ sung
│   └── combined_data.csv             # Dữ liệu đã xử lý
├── models/                     # Mô hình đã huấn luyện
│   ├── decision_tree_entropy.pkl
│   ├── decision_tree_gini.pkl
│   └── naive_bayes_best.pkl
├── outputs/                    # Kết quả đầu ra
│   ├── figures/                # Biểu đồ
│   ├── model_comparison.csv    # Bảng so sánh
│   └── predictions_2025.csv    # Dự đoán năm 2025
├── src/                        # Mã nguồn
│   ├── __init__.py
│   ├── data_loader.py          # Đọc dữ liệu từ Excel
│   ├── preprocessing.py        # Tiền xử lý dữ liệu
│   ├── decision_tree_model.py  # Mô hình Decision Tree
│   ├── naive_bayes_model.py    # Mô hình Naive Bayes
│   ├── visualization.py        # Visualization
│   └── main.py                 # Pipeline chính
├── notebooks/                  # Jupyter notebooks
│   └── analysis.ipynb
├── streamlit_app.py            # Demo web app
├── requirements.txt            # Dependencies
├── deep-research.md            # Tài liệu chi tiết đồ án
└── README.md                   # Hướng dẫn này
```

## Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (macOS/Linux)
source .venv/bin/activate

# Kích hoạt (Windows)
.venv\Scripts\activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Sử dụng

### Chạy Pipeline chính

```bash
# Từ thư mục gốc của project
python src/main.py
```

Pipeline sẽ thực hiện:
1. Load dữ liệu thực từ nso (2019-2024)
2. Tiền xử lý và tạo features cho ML
3. Chia dữ liệu train/test (75/25)
4. Huấn luyện Decision Tree (Entropy & Gini)
5. Huấn luyện Naive Bayes (với threshold optimization)
6. Đánh giá và so sánh các mô hình
7. Dự đoán xu hướng kết hôn năm 2025
8. Lưu kết quả và mô hình

### Chạy Demo Streamlit

```bash
# Từ thư mục gốc của project
streamlit run streamlit_app.py
```

## Kết quả

### Hiệu suất mô hình
| Mô hình | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| Decision Tree (Entropy) | 0.91 | 0.89 | 0.73 | 0.80 | 0.92 |
| Decision Tree (Gini) | 0.91 | 0.89 | 0.73 | 0.80 | 0.92 |
| Naive Bayes (Gaussian) | 0.86 | 0.76 | 0.59 | 0.67 | 0.85 |

### Feature Importance (Decision Tree)
1. **nam_2022**: 71% - Năm 2022 là năm có sự thay đổi lớn trong xu hướng kết hôn
2. **nam_2019**: 9% - Năm cơ sở trước COVID
3. **nam_2021**: 7% - Năm ảnh hưởng COVID
4. **tfr_level_Thấp**: 5% - Vùng có tỷ suất sinh thấp
5. **vung_kinh_te**: 4% - Vùng kinh tế

### Dự đoán 2025
- Đa số các tỉnh/thành (63/64) được dự đoán xu hướng **GIẢM** số cuộc kết hôn
- Vùng Nam có xác suất giảm cao nhất
- Vùng Bắc và Trung có xu hướng ổn định hơn
