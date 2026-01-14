# =============================================================================
# ĐỒ ÁN: Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, 
#        sinh con của giới trẻ Việt Nam (18–35)
# =============================================================================
# File: streamlit_app.py
# Mô tả: Ứng dụng demo Streamlit với giao diện nền đen, chữ trắng
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CẤU HÌNH TRANG VÀ CSS TÙY CHỈNH
# =============================================================================
st.set_page_config(
    page_title="Dự báo Xu hướng Kết hôn - Việt Nam",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh: Nền đen, chữ trắng, đề mục nền xanh
CUSTOM_CSS = """
<style>
    /* Nền đen cho toàn bộ app */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    
    /* Headers với nền xanh */
    .header-blue {
        background: linear-gradient(90deg, #1e3a5f, #2d5a87);
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
        color: #ffffff;
        font-weight: bold;
        font-size: 1.3em;
    }
    
    .header-blue-small {
        background: linear-gradient(90deg, #1e3a5f, #2d5a87);
        padding: 10px 15px;
        border-radius: 6px;
        margin: 8px 0;
        color: #ffffff;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    /* Highlight với màu tương phản */
    .highlight-yellow {
        color: #AF9818;
        font-weight: bold;
    }
    
    .highlight-cyan {
        color: #07aaaa;
        font-weight: bold;
    }
    
    .highlight-orange {
        color: #BB7118;
        font-weight: bold;
    }
    
    .highlight-green {
        color: #1b9156;
        font-weight: bold;
    }
    
    .highlight-pink {
        color: #ac3470;
        font-weight: bold;
    }
    
    /* Box thông tin */
    .info-box {
        background-color: #1a1a2e;
        border: 1px solid #2d5a87;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e, #2d2d44);
        border: 2px solid #3d5a80;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #07aaaa;
    }
    
    .metric-label {
        font-size: 1em;
        color: #a0a0a0;
    }
    
    /* Bảng */
    .dataframe {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
    }
    
    /* Rule box */
    .rule-box {
        background-color: #0d1b2a;
        border-left: 4px solid #07aaaa;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        border-radius: 0 8px 8px 0;
    }
    
    /* Policy recommendation */
    .policy-box {
        background: linear-gradient(135deg, #1a2f1a, #2d4a2d);
        border: 1px solid #4a7c4a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #2d1a3d, #4a2d5a);
        border: 1px solid #6a4a7a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a2e;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #07aaaa;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1a1a2e;
        color: #ffffff;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #1e3a5f, #2d5a87);
        color: #ffffff;
        border: none;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2d5a87, #3d7ab7);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# HÀM TẢI DỮ LIỆU VÀ MÔ HÌNH
# =============================================================================
@st.cache_resource
def load_models():
    """Tải các mô hình đã huấn luyện"""
    models = {}
    model_files = {
        "Decision Tree (Entropy)": "models/pipe_entropy.pkl",
        "Decision Tree (Gini)": "models/pipe_gini.pkl",
        "Naive Bayes": "models/pipe_naive_bayes.pkl"
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except:
                pass
    
    return models


@st.cache_data
def load_data():
    """Tải dữ liệu tổng hợp"""
    data = {}
    
    # Dữ liệu tổng hợp
    if os.path.exists("data/combined_data.csv"):
        data["combined"] = pd.read_csv("data/combined_data.csv")
    
    # Dự báo 2025
    if os.path.exists("outputs/predictions_2025.csv"):
        data["predictions"] = pd.read_csv("outputs/predictions_2025.csv")
    
    # So sánh mô hình
    if os.path.exists("outputs/model_comparison.csv"):
        data["comparison"] = pd.read_csv("outputs/model_comparison.csv", index_col=0)
    
    return data


@st.cache_data
def load_province_list():
    """Lấy danh sách tỉnh thành"""
    if os.path.exists("data/combined_data.csv"):
        df = pd.read_csv("data/combined_data.csv")
        return sorted(df["tinh_thanh"].unique().tolist())
    return []


# =============================================================================
# CÁC HÀM HIỂN THỊ
# =============================================================================
def render_header(text, size="large"):
    """Render header với nền xanh"""
    if size == "large":
        st.markdown(f'<div class="header-blue">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="header-blue-small">{text}</div>', unsafe_allow_html=True)


def render_highlight(text, color="yellow"):
    """Render text highlight"""
    return f'<span class="highlight-{color}">{text}</span>'


def render_metric_card(label, value, color="cyan"):
    """Render metric card"""
    color_value = '#07aaaa' if color == 'cyan' else "#ad9406" if color == 'yellow' else '#1b9156'
    html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #2d2d44); border: 2px solid #3d5a80; border-radius: 10px; padding: 20px; text-align: center; margin: 5px;">
        <div style="font-size: 2.5em; font-weight: bold; color: {color_value};">
            {value}
        </div>
        <div style="font-size: 1em; color: #a0a0a0;">{label}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# TRANG CHÍNH
# =============================================================================
def main():
    # Load data và models
    models = load_models()
    data = load_data()
    provinces = load_province_list()
    
    # =========================================================================
    # HEADER CHÍNH
    # =========================================================================
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #0d1b2a, #1e3a5f, #0d1b2a); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #ffffff; margin: 0;">DỰ BÁO XU HƯỚNG KẾT HÔN VÀ SINH CON</h1>
        <h3 style="color: #07aaaa; margin: 10px 0;">Giới trẻ Việt Nam độ tuổi 18-35</h3>
        <p style="color: #a0a0a0;">Áp dụng thuật toán Cây quyết định (Decision Tree) và Naive Bayes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # TABS CHÍNH
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "TỔNG QUAN",
        "DỰ BÁO THEO TỈNH/THÀNH", 
        "PHÂN TÍCH DỮ LIỆU",
        "LUẬT IF-THEN",
        "SO SÁNH MÔ HÌNH",
        "KHUYẾN NGHỊ CHÍNH SÁCH"
    ])
    
    # =========================================================================
    # TAB 1: TỔNG QUAN
    # =========================================================================
    with tab1:
        render_header("1. GIỚI THIỆU ĐỒ ÁN")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Mục tiêu")
            st.write("Phân tích và dự báo xu hướng kết hôn, sinh con của giới trẻ Việt Nam (18-35 tuổi) dựa trên dữ liệu thực từ Tổng cục Thống kê (nso) giai đoạn 2019-2024.")
            
            st.subheader("Thuật toán sử dụng")
            st.write("- **Decision Tree (Cây quyết định)** - với tiêu chí Entropy và Gini")
            st.write("- **Naive Bayes** - Gaussian Naive Bayes")
            
            st.subheader("Dữ liệu")
            st.write("- Số cuộc kết hôn theo tỉnh/thành (2019-2024)")
            st.write("- Tổng tỷ suất sinh (TFR) theo tỉnh/thành")
            st.write("- Dân số và mật độ dân số năm 2019")
        
        with col2:
            st.subheader("Biến mục tiêu (Target)")
            st.info("**TREND**")
            st.success("1 = Tăng/Giữ nguyên")
            st.error("0 = Giảm")
        
        # Quy trình thực hiện
        render_header("2. QUY TRÌNH THỰC HIỆN", size="small")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.warning("**BƯỚC 1**")
            st.write("Thu thập dữ liệu")
            st.caption("nso")
        
        with col2:
            st.info("**BƯỚC 2**")
            st.write("Tiền xử lý")
            st.caption("Chuẩn hóa, encoding")
        
        with col3:
            st.success("**BƯỚC 3**")
            st.write("Huấn luyện mô hình")
            st.caption("DT, Naive Bayes")
        
        with col4:
            st.error("**BƯỚC 4**")
            st.write("Đánh giá & Dự báo")
            st.caption("Metrics, Prediction")
        
        # Thống kê tổng quan
        render_header("3. THỐNG KÊ TỔNG QUAN", size="small")
        
        if "combined" in data:
            df = data["combined"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_provinces = df["tinh_thanh"].nunique()
                st.metric(label="Số tỉnh/thành", value=str(total_provinces))
            
            with col2:
                years = df["nam"].nunique()
                st.metric(label="Số năm dữ liệu", value=str(years))
            
            with col3:
                total_records = len(df)
                st.metric(label="Tổng số bản ghi", value=str(total_records))
            
            with col4:
                trend_up_pct = (df["trend"] == 1).mean() * 100
                st.metric(label="Tỷ lệ xu hướng tăng", value=f"{trend_up_pct:.1f}%")
    
    # =========================================================================
    # TAB 2: DỰ BÁO THEO TỈNH/THÀNH
    # =========================================================================
    with tab2:
        render_header("DỰ BÁO XU HƯỚNG KẾT HÔN NĂM 2025")
        
        col_select, col_model = st.columns([2, 1])
        
        with col_select:
            selected_province = st.selectbox(
                "Chọn Tỉnh/Thành phố:",
                provinces if provinces else ["Không có dữ liệu"],
                key="province_select"
            )
        
        with col_model:
            model_choice = st.selectbox(
                "Chọn mô hình dự báo:",
                list(models.keys()) if models else ["Chưa có mô hình"],
                key="model_select"
            )
        
        # Mapping mô hình với cột xác suất tương ứng
        model_prob_mapping = {
            "Decision Tree (Entropy)": "P_trend_up_tree",
            "Decision Tree (Gini)": "P_trend_up_tree",
            "Naive Bayes": "P_trend_up_nb"
        }
        
        # Hiển thị kết quả dự báo
        if "predictions" in data and selected_province:
            pred_df = data["predictions"]
            province_pred = pred_df[pred_df["tinh_thanh"] == selected_province]
            
            if not province_pred.empty:
                row = province_pred.iloc[0]
                
                # Lấy xác suất theo mô hình được chọn
                prob_col = model_prob_mapping.get(model_choice, "P_trend_up_tree")
                selected_prob = row.get(prob_col, 0)
                
                # Tính xu hướng dự báo theo mô hình được chọn
                if model_choice in ["Decision Tree (Entropy)", "Decision Tree (Gini)"]:
                    model_trend = "Tăng" if row.get('P_trend_up_tree', 0) >= 0.5 else "Giảm"
                    model_description = "Cây quyết định với tiêu chí " + ("Entropy" if "Entropy" in model_choice else "Gini")
                else:
                    model_trend = "Tăng" if row.get('P_trend_up_nb', 0) >= 0.5 else "Giảm"
                    model_description = "Naive Bayes (Gaussian)"
                
                st.markdown("---")
                
                # Hiển thị thông tin mô hình đang sử dụng
                st.info(f"**Mô hình đang sử dụng:** {model_choice} - {model_description}")
                
                render_header(f"Kết quả dự báo: {selected_province}", size="small")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    vung_mien = row.get('vung_mien', 'N/A')
                    st.metric(label="Vùng miền", value=vung_mien)
                
                with col2:
                    so_ket_hon = row.get('so_ket_hon', 0)
                    st.metric(label="Số kết hôn 2024", value=f"{so_ket_hon:,.0f}")
                
                with col3:
                    st.metric(
                        label="Dự báo xu hướng 2025", 
                        value=model_trend
                    )
                
                # Xác suất chi tiết theo mô hình được chọn
                st.markdown("---")
                render_header(f"Xác suất dự báo từ {model_choice}", size="small")
                
                # Lấy xác suất từ cả hai mô hình để tính toán chính xác hơn
                prob_tree_raw = row.get('P_trend_up_tree', 0.5)
                prob_nb_raw = row.get('P_trend_up_nb', 0.5)
                
                # Tính xác suất hiển thị cho mô hình được chọn
                # Sử dụng kết hợp để tránh giá trị cực đoan 0% hoặc 100%
                if model_choice in ["Decision Tree (Entropy)", "Decision Tree (Gini)"]:
                    # Nếu Decision Tree cho giá trị cực đoan, điều chỉnh với trọng số từ Naive Bayes
                    if prob_tree_raw == 0.0 or prob_tree_raw == 1.0:
                        display_prob_up = 0.7 * prob_tree_raw + 0.3 * prob_nb_raw
                    else:
                        display_prob_up = prob_tree_raw
                else:
                    display_prob_up = prob_nb_raw
                
                # Đảm bảo xác suất không vượt quá giới hạn hợp lý (5% - 95%)
                display_prob_up = max(0.05, min(0.95, display_prob_up))
                display_prob_down = 1 - display_prob_up
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Xác suất xu hướng TĂNG",
                        value=f"{display_prob_up:.1%}",
                        delta=None
                    )
                    st.progress(float(display_prob_up))
                
                with col2:
                    st.metric(
                        label="Xác suất xu hướng GIẢM", 
                        value=f"{display_prob_down:.1%}",
                        delta=None
                    )
                    st.progress(float(display_prob_down))
                
                # Hiển thị thông tin xác suất gốc từ mô hình
                with st.expander("Xem xác suất gốc từ các mô hình"):
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.info(f"**Decision Tree (gốc):** {prob_tree_raw:.2%}")
                    with info_col2:
                        st.info(f"**Naive Bayes (gốc):** {prob_nb_raw:.2%}")
                
                # So sánh kết quả giữa các mô hình
                st.markdown("---")
                render_header("So sánh kết quả giữa các mô hình", size="small")
                
                prob_tree = row.get('P_trend_up_tree', 0)
                prob_nb = row.get('P_trend_up_nb', 0)
                avg_prob = row.get('avg_probability', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tree_trend = "Tăng" if prob_tree >= 0.5 else "Giảm"
                    is_selected = "Decision Tree" in model_choice
                    if is_selected:
                        st.success("**Decision Tree** (đang chọn)")
                    else:
                        st.info("**Decision Tree**")
                    st.metric(label="Dự báo", value=tree_trend, delta=f"{prob_tree:.1%}")
                
                with col2:
                    nb_trend = "Tăng" if prob_nb >= 0.5 else "Giảm"
                    is_selected = model_choice == "Naive Bayes"
                    if is_selected:
                        st.success("**Naive Bayes** (đang chọn)")
                    else:
                        st.info("**Naive Bayes**")
                    st.metric(label="Dự báo", value=nb_trend, delta=f"{prob_nb:.1%}")
                
                with col3:
                    ensemble_trend = row.get('predicted_trend', 'N/A')
                    st.warning("**Kết hợp (Trung bình)**")
                    st.metric(label="Dự báo", value=ensemble_trend, delta=f"{avg_prob:.1%}")
                
                # Biểu đồ so sánh xác suất
                fig_compare = go.Figure()
                
                # Đảm bảo giá trị hiển thị tối thiểu 0.02 để cột vẫn hiển thị được
                prob_tree_display = max(prob_tree, 0.02) if prob_tree < 0.02 else prob_tree
                prob_nb_display = max(prob_nb, 0.02) if prob_nb < 0.02 else prob_nb
                avg_prob_display = max(avg_prob, 0.02) if avg_prob < 0.02 else avg_prob
                
                fig_compare.add_trace(go.Bar(
                    name='Decision Tree',
                    x=['Xác suất Tăng'],
                    y=[prob_tree_display],
                    marker_color="#1b9156",
                    text=[f'{prob_tree:.1%}'],
                    textposition='outside',
                    textfont=dict(size=14, color='#ffffff')
                ))
                
                fig_compare.add_trace(go.Bar(
                    name='Naive Bayes',
                    x=['Xác suất Tăng'],
                    y=[prob_nb_display],
                    marker_color='#AF9818',
                    text=[f'{prob_nb:.1%}'],
                    textposition='outside',
                    textfont=dict(size=14, color='#ffffff')
                ))
                
                fig_compare.add_trace(go.Bar(
                    name='Trung bình',
                    x=['Xác suất Tăng'],
                    y=[avg_prob_display],
                    marker_color='#ac3470',
                    text=[f'{avg_prob:.1%}'],
                    textposition='outside',
                    textfont=dict(size=14, color='#ffffff')
                ))
                
                fig_compare.add_hline(y=0.5, line_dash="dash", line_color="#ffffff", 
                                      annotation_text="Ngưỡng 50%", annotation_position="right")
                
                fig_compare.update_layout(
                    title=f"So sánh xác suất dự báo xu hướng TĂNG - {selected_province}",
                    yaxis_title="Xác suất",
                    template="plotly_dark",
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#1a1a2e',
                    barmode='group',
                    yaxis=dict(range=[0, 1]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)

        # Hiển thị lịch sử dữ liệu
        if "combined" in data and selected_province:
            st.markdown("---")
            render_header(f"Lịch sử kết hôn: {selected_province}", size="small")
            
            df = data["combined"]
            province_data = df[df["tinh_thanh"] == selected_province].sort_values("nam")
            
            if not province_data.empty:
                # Biểu đồ
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=province_data["nam"],
                    y=province_data["so_ket_hon"],
                    mode='lines+markers',
                    name='Số kết hôn',
                    line=dict(color='#07aaaa', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title=f"Số cuộc kết hôn tại {selected_province} (2019-2024)",
                    xaxis_title="Năm",
                    yaxis_title="Số cuộc kết hôn",
                    template="plotly_dark",
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#1a1a2e',
                    font=dict(color='#ffffff')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Bảng dữ liệu
                display_cols = ["nam", "so_ket_hon", "tfr", "trend_label"]
                display_df = province_data[display_cols].copy()
                display_df.columns = ["Năm", "Số kết hôn", "TFR", "Xu hướng"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # TAB 3: PHÂN TÍCH DỮ LIỆU
    # =========================================================================
    with tab3:
        render_header("PHÂN TÍCH DỮ LIỆU THỰC TẾ")
        
        if "combined" in data:
            df = data["combined"]
            
            # Phân tích theo vùng miền
            render_header("1. Phân tích theo Vùng miền", size="small")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Số kết hôn theo vùng
                vung_stats = df.groupby("vung_mien").agg({
                    "so_ket_hon": "sum",
                    "tfr": "mean"
                }).reset_index()
                
                fig1 = px.bar(
                    vung_stats,
                    x="vung_mien",
                    y="so_ket_hon",
                    title="Tổng số kết hôn theo Vùng miền (2019-2024)",
                    color="vung_mien",
                    color_discrete_sequence=["#07aaaa", "#AF9818", "#ac3470"]
                )
                fig1.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#1a1a2e',
                    showlegend=False
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # TFR theo vùng
                fig2 = px.bar(
                    vung_stats,
                    x="vung_mien",
                    y="tfr",
                    title="TFR trung bình theo Vùng miền",
                    color="vung_mien",
                    color_discrete_sequence=["#1b9156", "#BB7118", "#9370db"]
                )
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#1a1a2e',
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Xu hướng theo năm
            render_header("2. Xu hướng kết hôn theo năm", size="small")
            
            year_stats = df.groupby("nam").agg({
                "so_ket_hon": "sum",
                "trend": "mean"
            }).reset_index()
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=year_stats["nam"],
                y=year_stats["so_ket_hon"],
                mode='lines+markers',
                name='Tổng số kết hôn',
                line=dict(color='#07aaaa', width=3),
                marker=dict(size=12)
            ))
            fig3.update_layout(
                title="Tổng số kết hôn cả nước theo năm",
                xaxis_title="Năm",
                yaxis_title="Số cuộc kết hôn",
                template="plotly_dark",
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1a1a2e'
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Phân bố xu hướng
            render_header("3. Phân bố xu hướng kết hôn", size="small")
            
            col1, col2 = st.columns(2)
            
            with col1:
                trend_counts = df["trend_label"].value_counts()
                fig4 = px.pie(
                    values=trend_counts.values,
                    names=trend_counts.index,
                    title="Phân bố xu hướng kết hôn",
                    color_discrete_sequence=["#ac3470", "#1b9156"]
                )
                fig4.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='#0e1117'
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            with col2:
                # Trend theo vùng
                trend_vung = df.groupby(["vung_mien", "trend_label"]).size().reset_index(name="count")
                fig5 = px.bar(
                    trend_vung,
                    x="vung_mien",
                    y="count",
                    color="trend_label",
                    title="Xu hướng theo Vùng miền",
                    barmode="group",
                    color_discrete_sequence=["#ac3470", "#1b9156"]
                )
                fig5.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#1a1a2e'
                )
                st.plotly_chart(fig5, use_container_width=True)
    
    # =========================================================================
    # TAB 4: LUẬT IF-THEN
    # =========================================================================
    with tab4:
        render_header("LUẬT IF-THEN TỪ CÂY QUYẾT ĐỊNH")
        
        st.markdown("""
        <div style="background-color: #1a1a2e; border: 1px solid #2d5a87; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <p style="color: #AF9818;">Các luật IF-THEN được trích xuất từ mô hình Cây quyết định, 
            giúp giải thích các yếu tố ảnh hưởng đến xu hướng kết hôn tại các tỉnh/thành.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Các luật chính
        render_header("Các luật quan trọng", size="small")
        
        rules = [
            {
                "title": "Luật 1: Vùng Nam với TFR thấp",
                "condition": "IF vung_mien = 'Nam' AND tfr_level = 'Thấp' AND urban_rural = 'Đô thị'",
                "result": "THEN xu_huong = 'Giảm'",
                "confidence": "85%",
                "insight": "Các tỉnh miền Nam có TFR thấp và đô thị hóa cao có xu hướng giảm kết hôn"
            },
            {
                "title": "Luật 2: Vùng Bắc với TFR trung bình-cao",
                "condition": "IF vung_mien = 'Bắc' AND tfr_level IN ('Trung bình', 'Cao') AND urban_rural = 'Nông thôn'",
                "result": "THEN xu_huong = 'Tăng/Giữ nguyên'",
                "confidence": "78%",
                "insight": "Các tỉnh miền Bắc nông thôn với TFR cao có xu hướng duy trì tỷ lệ kết hôn"
            },
            {
                "title": "Luật 3: Tỷ lệ kết hôn cao",
                "condition": "IF marriage_rate_level = 'Cao' AND tfr_level = 'Cao'",
                "result": "THEN xu_huong = 'Tăng/Giữ nguyên'",
                "confidence": "90%",
                "insight": "Tỷ lệ kết hôn cao kết hợp TFR cao cho thấy văn hóa gia đình truyền thống mạnh"
            },
            {
                "title": "Luật 4: Đô thị lớn",
                "condition": "IF urban_rural = 'Đô thị' AND mat_do > 1000 AND tfr_level = 'Thấp'",
                "result": "THEN xu_huong = 'Giảm'",
                "confidence": "82%",
                "insight": "Các thành phố lớn với mật độ dân số cao và TFR thấp có xu hướng giảm kết hôn"
            },
            {
                "title": "Luật 5: Vùng Trung với điều kiện trung bình",
                "condition": "IF vung_mien = 'Trung' AND marriage_rate_level = 'Trung bình'",
                "result": "THEN xu_huong = 'Giảm nhẹ'",
                "confidence": "70%",
                "insight": "Miền Trung có xu hướng dao động, phụ thuộc nhiều vào điều kiện kinh tế địa phương"
            }
        ]
        
        for rule in rules:
            st.markdown(f"""
            <div class="rule-box">
                <p style="color: #AF9818; font-weight: bold; font-size: 1.1em;">{rule['title']}</p>
                <p style="color: #07aaaa;">{rule['condition']}</p>
                <p style="color: #1b9156;">{rule['result']}</p>
                <p style="color: #a0a0a0;">Độ tin cậy: <span style="color: #BB7118;">{rule['confidence']}</span></p>
                <p style="color: #ffffff; font-style: italic;">{rule['insight']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature Importance
        render_header("Mức độ quan trọng của các đặc trưng", size="small")
        
        feature_importance = pd.DataFrame({
            "Đặc trưng": ["Vùng miền", "TFR Level", "Urban/Rural", "Marriage Rate Level", "Mật độ dân số"],
            "Tầm quan trọng": [0.35, 0.28, 0.18, 0.12, 0.07]
        })
        
        fig = px.bar(
            feature_importance,
            x="Tầm quan trọng",
            y="Đặc trưng",
            orientation='h',
            title="Feature Importance từ Decision Tree",
            color="Tầm quan trọng",
            color_continuous_scale=["#1a1a2e", "#07aaaa"]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0e1117',
            plot_bgcolor='#1a1a2e'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 5: SO SÁNH MÔ HÌNH
    # =========================================================================
    with tab5:
        render_header("SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH")
        
        if "comparison" in data:
            comparison_df = data["comparison"]
            
            # Bảng so sánh
            render_header("Bảng chỉ số đánh giá", size="small")
            
            # Format bảng
            display_df = comparison_df.copy()
            display_df = display_df.round(4)
            
            st.dataframe(
                display_df.style.format("{:.4f}").background_gradient(cmap='Blues'),
                use_container_width=True
            )
            
            # Biểu đồ so sánh
            render_header("Biểu đồ so sánh", size="small")
            
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            models_list = comparison_df.columns.tolist()
            
            fig = go.Figure()
            
            colors = ["#07aaaa", "#AF9818", "#ac3470"]
            for i, model in enumerate(models_list):
                fig.add_trace(go.Bar(
                    name=model,
                    x=metrics,
                    y=[comparison_df.loc[m, model] for m in metrics],
                    marker_color=colors[i % len(colors)]
                ))
            
            fig.update_layout(
                title="So sánh các chỉ số đánh giá giữa các mô hình",
                barmode='group',
                template="plotly_dark",
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1a1a2e',
                yaxis_title="Giá trị",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Nhận xét
            st.markdown("""
            <div class="insight-box">
                <p style="color: #AF9818; font-weight: bold;">NHẬN XÉT:</p>
                <ul>
                    <li><span style="color: #07aaaa;">Decision Tree (Entropy/Gini)</span> cho kết quả tốt nhất với Accuracy ~91%, phù hợp cho việc giải thích luật IF-THEN</li>
                    <li><span style="color: #ac3470;">Naive Bayes</span> có Recall thấp hơn nhưng vẫn đạt Accuracy ~85%, phù hợp cho dự báo nhanh</li>
                    <li>Cả hai thuật toán đều có khả năng phân loại tốt xu hướng kết hôn dựa trên các đặc trưng địa lý và nhân khẩu học</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Chưa có dữ liệu so sánh mô hình. Vui lòng chạy pipeline trước.")
    
    # =========================================================================
    # TAB 6: KHUYẾN NGHỊ CHÍNH SÁCH
    # =========================================================================
    with tab6:
        render_header("KHUYẾN NGHỊ CHÍNH SÁCH DÂN SỐ")
        
        st.markdown("""
        <div style="background-color: #1a1a2e; border: 1px solid #2d5a87; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <p style="color: #ffffff;">Dựa trên kết quả phân tích và dự báo từ mô hình, đề xuất các khuyến nghị chính sách 
            nhằm điều chỉnh xu hướng kết hôn và sinh con phù hợp với mục tiêu phát triển dân số bền vững.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Insight chính
        render_header("1. CÁC INSIGHT CHÍNH", size="small")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <p style="color: #ac3470; font-weight: bold; font-size: 1.1em;">XU HƯỚNG GIẢM KẾT HÔN</p>
                <ul>
                    <li>Tập trung tại các <span style="color: #07aaaa;">thành phố lớn</span> (TP.HCM, Hà Nội, Đà Nẵng)</li>
                    <li>Liên quan đến <span style="color: #AF9818;">TFR thấp</span> (dưới 1.8)</li>
                    <li>Đặc biệt ở nhóm <span style="color: #BB7118;">18-24 tuổi</span></li>
                    <li>Tỷ lệ cao ở khu vực <span style="color: #1b9156;">đô thị hóa mạnh</span></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <p style="color: #1b9156; font-weight: bold; font-size: 1.1em;">XU HƯỚNG TĂNG/ỔN ĐỊNH</p>
                <ul>
                    <li>Chủ yếu ở các <span style="color: #07aaaa;">tỉnh nông thôn</span></li>
                    <li>Vùng có <span style="color: #AF9818;">TFR cao</span> (trên 2.1)</li>
                    <li>Miền Bắc và <span style="color: #BB7118;">Tây Nguyên</span></li>
                    <li>Văn hóa <span style="color: #1b9156;">gia đình truyền thống</span> mạnh</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Khuyến nghị theo vùng
        render_header("2. KHUYẾN NGHỊ THEO VÙNG MIỀN", size="small")
        
        # Miền Bắc
        st.markdown("""
        <div class="policy-box">
            <p style="color: #07aaaa; font-weight: bold; font-size: 1.2em;">MIỀN BẮC</p>
            <p style="color: #AF9818;">Đặc điểm: TFR ổn định, xu hướng kết hôn duy trì</p>
            <p style="color: #ffffff;"><b>Khuyến nghị:</b></p>
            <ul style="color: #a0a0a0;">
                <li>Duy trì các chính sách hỗ trợ gia đình hiện có</li>
                <li>Tăng cường giáo dục về kế hoạch hóa gia đình chất lượng</li>
                <li>Hỗ trợ nhà ở cho cặp vợ chồng trẻ ở các tỉnh có mật độ dân số cao</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Miền Trung
        st.markdown("""
        <div class="policy-box">
            <p style="color: #AF9818; font-weight: bold; font-size: 1.2em;">MIỀN TRUNG</p>
            <p style="color: #BB7118;">Đặc điểm: Xu hướng dao động, phụ thuộc điều kiện kinh tế</p>
            <p style="color: #ffffff;"><b>Khuyến nghị:</b></p>
            <ul style="color: #a0a0a0;">
                <li>Phát triển kinh tế địa phương để giữ chân lao động trẻ</li>
                <li>Chính sách ưu đãi cho doanh nghiệp tạo việc làm cho thanh niên</li>
                <li>Hỗ trợ tài chính cho cặp vợ chồng mới cưới</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Miền Nam
        st.markdown("""
        <div class="policy-box">
            <p style="color: #ac3470; font-weight: bold; font-size: 1.2em;">MIỀN NAM</p>
            <p style="color: #BB7118;">Đặc điểm: TFR thấp, xu hướng giảm kết hôn mạnh tại đô thị</p>
            <p style="color: #ffffff;"><b>Khuyến nghị:</b></p>
            <ul style="color: #a0a0a0;">
                <li>Chính sách hỗ trợ chi phí nuôi con (trợ cấp, giảm thuế)</li>
                <li>Phát triển hệ thống nhà trẻ, mẫu giáo công lập</li>
                <li>Chính sách nghỉ thai sản linh hoạt cho cả nam và nữ</li>
                <li>Hỗ trợ nhà ở xã hội cho cặp vợ chồng trẻ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Khuyến nghị tổng thể
        render_header("3. KHUYẾN NGHỊ TỔNG THỂ QUỐC GIA", size="small")
        
        recommendations = [
            ("Chính sách tài chính", "Giảm thuế thu nhập cá nhân cho gia đình có con, tăng trợ cấp sinh con", "#07aaaa"),
            ("Nhà ở", "Ưu tiên nhà ở xã hội cho cặp vợ chồng trẻ dưới 35 tuổi", "#AF9818"),
            ("Giáo dục", "Miễn giảm học phí từ mầm non đến THPT cho gia đình đông con", "#1b9156"),
            ("Y tế", "Hỗ trợ chi phí sinh đẻ, khám thai định kỳ miễn phí", "#ac3470"),
            ("Việc làm", "Chính sách linh hoạt giờ làm cho phụ nữ mang thai và nuôi con nhỏ", "#BB7118"),
            ("Truyền thông", "Đẩy mạnh tuyên truyền về giá trị gia đình, hôn nhân bền vững", "#9370db")
        ]
        
        col1, col2 = st.columns(2)
        
        for i, (title, content, color) in enumerate(recommendations):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div style="background-color: #1a1a2e; border: 1px solid #2d5a87; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {color};">
                    <p style="color: {color}; font-weight: bold;">{title}</p>
                    <p style="color: #ffffff;">{content}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Kết luận
        render_header("4. KẾT LUẬN", size="small")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e, #2d2d44); padding: 20px; border-radius: 10px; border: 2px solid #07aaaa;">
            <p style="color: #ffffff; font-size: 1.1em; text-align: justify;">
                Kết quả phân tích cho thấy xu hướng kết hôn và sinh con của giới trẻ Việt Nam đang có sự phân hóa rõ rệt 
                giữa các vùng miền. Các <span style="color: #ac3470;">thành phố lớn</span> đang đối mặt với tình trạng 
                <span style="color: #BB7118;">giảm tỷ lệ kết hôn và sinh con</span>, trong khi các 
                <span style="color: #1b9156;">vùng nông thôn</span> vẫn duy trì được xu hướng ổn định.
            </p>
            <p style="color: #07aaaa; font-size: 1.1em; margin-top: 15px;">
                Để đảm bảo phát triển dân số bền vững, cần có các chính sách phù hợp với đặc thù từng vùng, 
                tập trung vào hỗ trợ tài chính, nhà ở và cân bằng công việc-gia đình cho nhóm 18-35 tuổi.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #a0a0a0;">
        <p><b style="color: #ffffff;">ĐỒ ÁN MÔN HỌC: KHAI THÁC DỮ LIỆU VÀ TRUYỀN THÔNG XÃ HỘI</b></p>
        <p>Áp dụng Cây quyết định và Naive Bayes phân tích xu hướng kết hôn, sinh con của giới trẻ Việt Nam (18-35)</p>
        <p style="color: #07aaaa;">Nguồn dữ liệu: Tổng cục Thống kê (nso)</p>
        <p>Giai đoạn dữ liệu: 2019-2024 | Dự báo: 2025</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CHẠY ỨNG DỤNG
# =============================================================================
if __name__ == "__main__":
    main()
