# Bộ đồ án: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, sinh con của giới trẻ Việt Nam (18–35)

Mục tiêu chính: dự báo **marital_class** (0: chưa kết hôn; 1: đã/đang chung sống/kết hôn) dựa trên dữ liệu synthetic microdata (≥1.500 cá nhân), tham số hóa theo thông tin Tổng cục Thống kê (GSO).

---

## A. Kết quả cốt lõi

- **File CSV synthetic**: ≥1.500 dòng, mỗi dòng 1 cá nhân 18–35 với các cột  
  `age_group`, `sex`, `region` (Bắc/Trung/Nam), `urban_rural`, `education`, `income_pc`, `income_level`, `has_child`, `marital_class`.  
- **Notebook / script Python**: pipeline tiền xử lý → Decision Tree (entropy/gini) → Naive Bayes (CategoricalNB) → đánh giá (Accuracy / Precision / Recall / F1 / ROC-AUC) → rút luật IF–THEN → posterior cho personas.  
- **Bộ biểu đồ**: ROC curve, confusion matrix, biểu đồ mô tả phân phối.  
- **Dàn ý báo cáo** (5–6 chương) theo quy trình KDD.

---

## 1. Chọn và hiểu dữ liệu (Selection / Understanding)

Nguồn tham chiếu để tham số hóa synthetic microdata:  

- Tổng cục Thống kê (GSO) – Trang Dân số: Tổng hợp các chỉ tiêu dân số Việt Nam. [gso.gov.vn](https://www.gso.gov.vn/dan-so/)  
- GSO – Diện tích, dân số và mật độ dân số phân theo địa phương. [gso.gov.vn](https://www.gso.gov.vn/px-web-2/?pxid=V0201&theme=D%C3%A2n%20s%E1%BB%91%20v%C3%A0%20lao%20%C4%91%E1%BB%99ng)  
- GSO – Số cuộc kết hôn phân theo địa phương. [gso.gov.vn](https://www.gso.gov.vn/px-web-2/?pxid=V0228&theme=D%C3%A2n%20s%E1%BB%91%20v%C3%A0%20lao%20%C4%91%E1%BB%99ng)  
- GSO – Tổng tỷ suất sinh (TFR) phân theo địa phương. [gso.gov.vn](https://www.gso.gov.vn/px-web-2/?pxid=V0216&theme=D%C3%A2n%20s%E1%BB%91%20v%C3%A0%20lao%20%C4%91%E1%BB%99ng)

---

## 2. Đề xuất cấu trúc bảng dữ liệu cuối (Integration)

**Các biến bắt buộc:**

| Biến            | Kiểu dữ liệu | Giá trị / Phân loại                          |
|-----------------|--------------|-----------------------------------------------|
| age_group       | Categorical  | 18–24, 25–29, 30–35                           |
| sex             | Categorical  | Nam, Nữ                                       |
| region          | Categorical  | Bắc, Trung, Nam                               |
| urban_rural     | Categorical  | Đô thị, Nông thôn                            |
| education       | Categorical  | ≤THPT, CĐ, ĐH, >ĐH                            |
| income_pc       | Float        | Thu nhập bình quân đầu người (lognormal)      |
| income_level    | Categorical  | low (≤P33), mid (P33–P66), high (>P66)         |
| has_child       | Binary       | 0, 1                                          |
| marital_class   | Binary *Target A* | 0: chưa kết hôn; 1: đã/đang kết hôn/chung sống |

**Lý do chọn biến**  
- **Tuổi**: quyết định mạnh hành vi kết hôn.  
- **Đô thị / Học vấn**: mức sinh/kết hôn muộn hơn ở đô thị và nhóm học vấn cao.  
- **Vùng**: Đông Nam Bộ (đại diện “Nam”) có mức sinh thấp.  
- Các dấu hiệu nhất quán với GSO.

---

## 3. Tiền xử lý dữ liệu (Preprocessing)

1. **Lọc** nhóm tuổi 18–35; tạo `age_group` theo bins `[18–24, 25–29, 30–35]`.  
2. **Thu nhập**: lognormal theo học vấn & đô thị; phân vị nội bộ P33/P66 để gán `income_level`.  
3. **Mã hóa**:  
   - Decision Tree: One-Hot Encoding (OHE).  
   - Naive Bayes: OrdinalEncoder (unknown_value=-1).  
4. **Chia tập** train/test = 75/25, stratify theo `marital_class`.  
5. Khi có CSV thực, chỉ cần map tên cột theo schema và chạy lại pipeline.

---

## 4. Tạo microdata tổng hợp tham số hóa (Python 3.11+)

Nguyên tắc: sinh N hồ sơ theo tỉ trọng vùng (45% Bắc, 20% Trung, 35% Nam), đô thị ≈44%; gán `has_child`, `marital_class` theo xác suất logistic phản ánh dấu từ GSO.

```python
# Code 1: Sinh dữ liệu + chuẩn hóa
import numpy as np, pandas as pd
np.random.seed(2026)
N = 3000
# 1) Khung dữ liệu danh mục
age = np.random.randint(18, 36, N)
sex = np.random.choice(["Nam","Nữ"], N, p=[0.49, 0.51])
region = np.random.choice(["Bắc","Trung","Nam"], N, p=[0.45,0.20,0.35])
urban_rural = np.random.choice(["Đô thị","Nông thôn"], N, p=[0.44,0.56])
education = np.random.choice(["≤THPT","CĐ","ĐH",">ĐH"], N, p=[0.48,0.20,0.28,0.04])
df = pd.DataFrame({"age":age,"sex":sex,"region":region,"urban_rural":urban_rural,"education":education})

# 2) Thu nhập lognormal
base = np.random.lognormal(mean=8.05, sigma=0.55, size=N)
edu_mult = df["education"].map({"≤THPT":0.85,"CĐ":1.0,"ĐH":1.18,">ĐH":1.35})
urb_mult = np.where(df["urban_rural"]=="Đô thị", 1.10, 0.95)
df["income_pc"] = base * edu_mult * urb_mult

# 3) age_group & income_level
df["age_group"] = pd.cut(df["age"], bins=[18,24,29,35],
    labels=["18-24","25-29","30-35"], include_lowest=True)
q33, q66 = df["income_pc"].quantile([0.33,0.66])
df["income_level"] = df["income_pc"].apply(
    lambda x: "low" if x<q33 else ("mid" if x<q66 else "high")
)

# 4) Xác suất logistic tham số hóa
def sigmoid(z): return 1/(1+np.exp(-z))
def logit_married(r):
    base = -2.2 + {"18-24":-0.6,"25-29":0.6,"30-35":1.0}[r.age_group]
    base += -0.25 if r.urban_rural=="Đô thị" else 0.25
    base += {"≤THPT":0.25,"CĐ":0.05,"ĐH":-0.20,">ĐH":-0.35}[r.education]
    base += -0.20 if r.region=="Nam" else (0.05 if r.region=="Bắc" else 0.0)
    return base
def logit_child(r):
    base = -2.8 + {"18-24":-0.7,"25-29":0.6,"30-35":1.2}[r.age_group]
    base += -0.35 if r.urban_rural=="Đô thị" else 0.35
    base += {"≤THPT":0.30,"CĐ":0.05,"ĐH":-0.25,">ĐH":-0.40}[r.education]
    base += -0.15 if r.region=="Nam" else (0.05 if r.region=="Bắc" else 0.0)
    return base

p_married = df.apply(logit_married, axis=1).pipe(sigmoid)
p_child   = df.apply(logit_child, axis=1).pipe(sigmoid)
df["marital_class"] = (np.random.rand(N) < p_married).astype(int)
df["has_child"]     = (np.random.rand(N) < p_child).astype(int)

# 5) Chọn cột kết quả
df = df[["age","age_group","sex","region","urban_rural","education",
         "income_pc","income_level","has_child","marital_class"]]
print(df.head(), df.shape)
```

---

## 5. Mô hình Cây quyết định (Decision Tree)

### 5.1. Cơ sở lý thuyết (theo slide Bai5)

- **Entropy**: H(S) = −∑ pᵢ · log₂ pᵢ  
- **Information Gain**: Gain(S,A) = H(S) − ∑ᵥ (|Sᵥ|/|S|)·H(Sᵥ)  
- **Gini**: Gini(S) = 1 − ∑ pᵢ²  

### 5.2. Thực thi & đánh giá

```python
# Code 2: Train + đánh giá Decision Tree (entropy/gini)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, RocCurveDisplay

features = ["age_group","sex","region","urban_rural","education","income_level","has_child"]
target = "marital_class"

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.25, stratify=df[target], random_state=2026
)
pre_tree = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown="ignore"), features)], remainder="drop")

results = {}
for crit in ["entropy","gini"]:
    clf = DecisionTreeClassifier(criterion=crit, max_depth=5, min_samples_split=60,
                                 class_weight="balanced", random_state=2026)
    pipe = Pipeline([("prep", pre_tree), ("model", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_proba)
    results[f"Tree-{crit}"] = {"acc":acc,"pr":pr,"rc":rc,"f1":f1,"auc":auc,"pipe":pipe}
    print(f"[Tree-{crit}] Acc={acc:.3f} Prec={pr:.3f} Rec={rc:.3f} F1={f1:.3f} AUC={auc:.3f}")
    print(confusion_matrix(y_test, y_pred))

# Xuất luật IF–THEN (mô hình entropy)
best = results["Tree-entropy"]["pipe"]
ohe_names = best.named_steps["prep"].named_transformers_["ohe"].get_feature_names_out(features)
print(export_text(best.named_steps["model"], feature_names=ohe_names.tolist()))

# Vẽ ROC & Confusion Matrix
import matplotlib.pyplot as plt, seaborn as sns
plt.figure(figsize=(6,5))
for name, res in results.items():
    RocCurveDisplay.from_estimator(res["pipe"], X_test, y_test, name=name)
plt.plot([0,1],[0,1],'k--'); plt.title("ROC Curves"); plt.tight_layout(); plt.show()
cm = confusion_matrix(y_test, best.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree (entropy) - Confusion Matrix"); plt.show()
```

**Ví dụ luật IF–THEN**  
- IF `age_group=18–24` AND `education`∈{ĐH,>ĐH} AND `urban_rural=Đô thị` AND `has_child=0` → dự báo “chưa kết hôn”.  
- IF `age_group=30–35` AND `urban_rural=Nông thôn` AND (`education≤THPT` OR `has_child=1`) → dự báo “đã kết hôn”.  
Những luật này phản ánh khác biệt đô thị–nông thôn, học vấn và tuổi [vietnam.unfpa.org](https://vietnam.unfpa.org/en/news/results-depth-analysis-2019-viet-nam-population-and-housing-census?utm_source=openai).

---

## 6. Mô hình Naive Bayes (CategoricalNB)

### 6.1. Cơ sở lý thuyết (theo slide Bai4)

- **Định lý Bayes**: P(Y|X) = P(Y)·∏ P(Xⱼ|Y) / P(X)  
- **Giả thiết** độc lập điều kiện giữa các thuộc tính Xⱼ khi biết Y  
- **Laplace smoothing** α: P̂(x|y) = (count(x,y)+α)/(count(y)+α·K)  

### 6.2. Thực thi & posterior cho personas

```python
# Code 3: Train + đánh giá Naive Bayes
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

pre_nb = ColumnTransformer([
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), features)
], remainder="drop")
pipe_nb = Pipeline([("prep", pre_nb), ("model", CategoricalNB(alpha=1.0))])

pipe_nb.fit(X_train, y_train)
y_pred_nb = pipe_nb.predict(X_test)
y_proba_nb = pipe_nb.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred_nb)
pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred_nb, average="binary")
auc = roc_auc_score(y_test, y_proba_nb)
print(f"[NaiveBayes] Acc={acc:.3f} Prec={pr:.3f} Rec={rc:.3f} F1={f1:.3f} AUC={auc:.3f}")
print(confusion_matrix(y_test, y_pred_nb))

# Personas
personas = pd.DataFrame({
    "age_group":["25-29","25-29","30-35","18-24","30-35"],
    "sex":["Nữ","Nam","Nữ","Nữ","Nam"],
    "region":["Nam","Bắc","Bắc","Trung","Nam"],
    "urban_rural":["Đô thị","Đô thị","Nông thôn","Đô thị","Nông thôn"],
    "education":["ĐH","ĐH","≤THPT",">ĐH","≤THPT"],
    "income_level":["mid","mid","low","high","low"],
    "has_child":[0,0,1,0,1]
})
post = pipe_nb.predict_proba(personas)[:,1]
print(pd.concat([personas, pd.Series(post, name="P_married")], axis=1))
```

---

## 7. Đánh giá & so sánh (Evaluation)

- **Chỉ số báo cáo**: Accuracy; Precision/Recall/F1 lớp 1; ROC-AUC.  
- **Decision Tree**: thường F1/ROC-AUC cao hơn, luật rõ ràng; cần kiểm soát overfitting (max_depth, min_samples_split).  
- **Naive Bayes**: nhanh, posterior trực tiếp cho personas; nhạy với vi phạm độc lập điều kiện (tuổi×học vấn, đô thị×học vấn).  
- Có thể lặp lại với random_state khác hoặc 5-fold để kiểm tra độ bền.

---

## 8. Insight chính và “So what?” (Synthesis)

- **Biến quan trọng nhất**: `age_group`, kế tiếp `urban_rural` và `education` → phản ánh xu hướng kết hôn/sinh con muộn hơn ở đô thị và nhóm học vấn cao. [vietnam.unfpa.org](https://vietnam.unfpa.org/en/news/results-population-and-housing-census-2019?utm_source=openai)  
- **Phân khúc có nguy cơ chưa kết hôn cao**:  
  - 18–24, đô thị, học vấn ĐH/>ĐH, chưa có con  
  - 25–29, đô thị, học vấn ĐH, thu nhập mid/high  
  - Vùng Nam (Đông Nam Bộ) mức chung thấp. [vietnam.unfpa.org](https://vietnam.unfpa.org/en/news/results-depth-analysis-2019-viet-nam-population-and-housing-census?utm_source=openai)  
- **Hàm ý chính sách**:  
  - Hỗ trợ nhà ở, dịch vụ giữ trẻ, môi trường làm việc thân thiện gia đình cho nhóm đô thị/học vấn cao.  
  - Truyền thông bình đẳng giới, kế hoạch hóa gia đình có trách nhiệm, tập trung đô thị/Đông Nam Bộ. [vietnam.unfpa.org](https://vietnam.unfpa.org/en/publications/viet-nam-population-projection-period-2019-2069)

---

## 9. Ứng dụng demo Streamlit (tùy chọn)

```python
# streamlit_app.py
import streamlit as st, pandas as pd, joblib
st.title("Dự báo xác suất kết hôn (18–35) • Decision Tree")
pipe = joblib.load("pipe_entropy.pkl")
# Inputs
age_group = st.selectbox("Nhóm tuổi", ["18-24","25-29","30-35"])
sex = st.selectbox("Giới tính", ["Nam","Nữ"])
region = st.selectbox("Vùng", ["Bắc","Trung","Nam"])
urban_rural = st.selectbox("Khu vực", ["Đô thị","Nông thôn"])
education = st.selectbox("Học vấn", ["≤THPT","CĐ","ĐH",">ĐH"])
income_level = st.selectbox("Thu nhập", ["low","mid","high"])
has_child = st.selectbox("Đã có con?", [0,1])
X = pd.DataFrame([{
    "age_group":age_group,"sex":sex,"region":region,"urban_rural":urban_rural,
    "education":education,"income_level":income_level,"has_child":int(has_child)
}])
proba = pipe.predict_proba(X)[0,1]
st.metric("P(marital_class=1)", f"{proba:.2%}")
```

---

## 10. Dàn ý báo cáo (5–6 chương)

1. **Giới thiệu**  
   - Bài toán: dự báo P(marital_class=1|X) và rút luật IF–THEN cho chính sách.  
   - Đóng góp: microdata synthetic, mô hình Cây/NB, personas, hàm ý chính sách.  
2. **Cơ sở lý thuyết**  
   - Decision Tree (Entropy, Gain, Gini, overfitting).  
   - Naive Bayes (Định lý Bayes, độc lập điều kiện, Laplace smoothing).  
3. **Dữ liệu & Tiền xử lý**  
   - Nguồn GSO, tham số hóa synthetic, schema, mã hóa, mô tả mẫu (bảng/biểu đồ).  
4. **Mô hình & Kết quả**  
   - Cấu hình, chỉ số (Accuracy/Precision/Recall/F1/ROC-AUC), ROC, confusion.  
   - ≥8 luật IF–THEN tiêu biểu; posterior cho ≥5 personas.  
5. **Thảo luận & Hàm ý chính sách**  
   - Đối chiếu với xu hướng đô thị–học vấn–vùng; phân khúc trọng điểm; hạn chế (synthetic, giả thiết NB).  
6. **Kết luận & Hướng mở**  
   - Nâng cấp: dùng CSV thật (VHLSS/TĐT), mở rộng Random Forest, Calibrated NB, kiểm định độ bền theo năm/vùng.

**Bảng & Biểu đồ nên có**  
- Bảng mô tả mẫu theo age_group/sex/urban_rural/education.  
- Biểu đồ tỷ lệ `has_child` theo `education × urban_rural`.  
- ROC so sánh Tree-Entropy/Tree-Gini/NB; ma trận nhầm lẫn.  
- Bảng luật IF–THEN; bảng posterior cho personas.

---

## Ghi chú xác thực nguồn

- **Tổng cục Thống kê (GSO) – Trang Dân số**: Tổng hợp các chỉ tiêu dân số Việt Nam. [gso.gov.vn](https://www.gso.gov.vn/dan-so/)  
- **GSO – Diện tích, dân số và mật độ dân số phân theo địa phương**: Dữ liệu dân số chi tiết theo 63 tỉnh/thành. [gso.gov.vn](https://www.gso.gov.vn/px-web-2/?pxid=V0201&theme=D%C3%A2n%20s%E1%BB%91%20v%C3%A0%20lao%20%C4%91%E1%BB%99ng)  
- **GSO – Tổng tỷ suất sinh (TFR) phân theo địa phương**: TFR toàn quốc và theo tỉnh/thành giai đoạn 2019-2024. [gso.gov.vn](https://www.gso.gov.vn/px-web-2/?pxid=V0216&theme=D%C3%A2n%20s%E1%BB%91%20v%C3%A0%20lao%20%C4%91%E1%BB%99ng)  
- **GSO – Số cuộc kết hôn phân theo địa phương**: Thống kê số cuộc kết hôn theo 63 tỉnh/thành giai đoạn 2019-2024. [gso.gov.vn](https://www.gso.gov.vn/px-web-2/?pxid=V0228&theme=D%C3%A2n%20s%E1%BB%91%20v%C3%A0%20lao%20%C4%91%E1%BB%99ng)  

---

## Lưu ý khi sử dụng với CSV thực

- Đọc CSV; đổi tên cột khớp schema; tạo `age_group`/`income_level`; chạy lại quy trình từ Bước 5.  
- Mốc thời điểm dữ liệu: Tổng cục Thống kê (GSO) giai đoạn 2019-2024.  
- Các con số (TFR, số cuộc kết hôn, dân số theo vùng) rút từ GSO 2019–2024.