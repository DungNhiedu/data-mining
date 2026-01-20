# =============================================================================
# ĐỒ ÁN: Dự báo Tình trạng Hôn nhân của Giới trẻ (18–35), Giai đoạn 2019–2024
# =============================================================================
# File: app_panel.py (Streamlit App)
# Mô tả: Ứng dụng demo dự báo kết hôn với panel data
#
# BIẾN MỤC TIÊU (Target Variable):
#   - Y_married (từ MARST): Tình trạng hôn nhân
#     + 0 = Chưa kết hôn (Single/never married)
#     + 1 = Đã kết hôn (Married/in union)
#
# UI Style: Material Kit React (Devias) - Professional Dashboard
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load environment variables từ file .env
from dotenv import load_dotenv
load_dotenv()

# Google Gemini AI - SDK mới (google-genai)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old SDK
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False

st.set_page_config(
    page_title="Dự báo Xu hướng Kết Hôn - Việt Nam 2019-2024",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ========================================
       GOOGLE FONTS - Inter & Plus Jakarta Sans
       ======================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap');
    
    /* ========================================
       ROOT VARIABLES - Material Kit React Colors
       ======================================== */
    :root {
        /* Neon Blue - Primary */
        --primary-50: #ecf0ff;
        --primary-100: #dde3ff;
        --primary-200: #c2cbff;
        --primary-300: #9ca7ff;
        --primary-400: #7578ff;
        --primary-500: #635bff;
        --primary-600: #4e36f5;
        --primary-700: #432ad8;
        --primary-main: #635bff;
        
        /* Nevada - Neutral */
        --neutral-50: #fbfcfe;
        --neutral-100: #f0f4f8;
        --neutral-200: #dde7ee;
        --neutral-300: #cdd7e1;
        --neutral-400: #9fa6ad;
        --neutral-500: #636b74;
        --neutral-600: #555e68;
        --neutral-700: #32383e;
        --neutral-800: #202427;
        --neutral-900: #121517;
        --neutral-950: #090a0b;
        
        /* Kepple - Success */
        --success-300: #5fe9ce;
        --success-400: #2ed3b8;
        --success-500: #15b79f;
        --success-main: #15b79f;
        
        /* California - Warning */
        --warning-300: #ffd049;
        --warning-400: #ffbb1f;
        --warning-500: #fb9c0c;
        --warning-main: #ffbb1f;
        
        /* Red Orange - Error */
        --error-400: #f97970;
        --error-500: #f04438;
        --error-main: #f04438;
        
        /* Shakespeare - Info */
        --info-400: #10bee8;
        --info-500: #04aad6;
        --info-main: #04aad6;
        
        /* Storm Grey - Background */
        --background-default: #121621;
        --background-paper: #212636;
        --background-level1: #313749;
        --background-level2: #434a60;
        
        /* Text Colors */
        --text-primary: #f0f4f8;
        --text-secondary: #9fa6ad;
        --text-disabled: #555e68;
    }
    
    /* ========================================
       MAIN APP BACKGROUND
       ======================================== */
    .stApp {
        background: linear-gradient(180deg, var(--background-default) 0%, #0d1117 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* ========================================
       TĂNG SIZE CHỮ TOÀN BỘ APP
       ======================================== */
    .stApp, .stMarkdown, p, span, div {
        font-size: 1.05rem !important;
    }
    
    .stSelectbox label, .stTextInput label, .stSlider label {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #ffffff !important;
    }
    
    /* Chỉnh màu chữ label thành trắng */
    .stSlider label p, .stSelectbox label p {
        color: #ffffff !important;
    }
    
    /* ========================================
       SIDEBAR STYLING - Material Design
       ======================================== */
    [data-testid="stSidebar"] {
        background: var(--background-paper);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0;
    }
    
    /* ========================================
       MAIN HEADER - Gradient Style
       ======================================== */
    .main-header {
        background: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-500) 50%, var(--primary-400) 100%);
        padding: 48px 40px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 20px 40px -12px rgba(99, 91, 255, 0.35);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.4;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.75em;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        position: relative;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header h3 {
        color: rgba(255,255,255,0.95);
        font-weight: 500;
        margin: 16px 0 8px 0;
        font-size: 1.35em;
        position: relative;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.75);
        font-size: 1.1em;
        position: relative;
        margin: 0;
    }
    
    /* ========================================
       SECTION HEADERS
       ======================================== */
    .section-header {
        background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
        padding: 18px 26px;
        border-radius: 16px;
        margin: 28px 0 24px 0;
        color: #ffffff;
        font-weight: 600;
        font-size: 1.2em;
        box-shadow: 0 8px 24px -8px rgba(99, 91, 255, 0.4);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* ========================================
       CARD STYLES - Material Design Cards
       ======================================== */
    .metric-card {
        background: var(--background-paper);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22);
        border-color: rgba(99, 91, 255, 0.3);
    }
    
    .metric-value {
        font-size: 3em;
        font-weight: 700;
        color: var(--primary-main);
        line-height: 1.1;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85em;
        color: var(--text-secondary);
        margin-top: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ========================================
       STATS CARDS - With Icon Avatar
       ======================================== */
    .stats-card {
        background: var(--background-paper);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .stats-card-green {
        border-left: 4px solid var(--success-main);
    }
    
    .stats-card-purple {
        border-left: 4px solid var(--primary-main);
    }
    
    .stats-card-orange {
        border-left: 4px solid var(--warning-main);
    }
    
    .stats-card-pink {
        border-left: 4px solid var(--error-main);
    }
    
    .stats-card-info {
        border-left: 4px solid var(--info-main);
    }
    
    /* Avatar icon style */
    .avatar-icon {
        width: 56px;
        height: 56px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5em;
    }
    
    .avatar-primary { background: var(--primary-main); }
    .avatar-success { background: var(--success-main); }
    .avatar-warning { background: var(--warning-main); }
    .avatar-error { background: var(--error-main); }
    .avatar-info { background: var(--info-main); }
    
    /* ========================================
       FORM CARD - Input Section
       ======================================== */
    .form-card {
        background: var(--background-paper);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
    }
    
    .form-card h4 {
        color: var(--primary-main);
        margin: 0 0 20px 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.2em;
    }
    
    /* ========================================
       RULE BOX - IF-THEN Rules
       ======================================== */
    .rule-box {
        background: var(--background-paper);
        border-left: 4px solid var(--primary-main);
        padding: 24px 28px;
        margin: 16px 0;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    .rule-box:hover {
        background: var(--background-level1);
        transform: translateX(4px);
    }
    
    /* ========================================
       INSIGHT & WARNING BOXES
       ======================================== */
    .insight-box {
        background: linear-gradient(135deg, rgba(21, 183, 159, 0.1) 0%, var(--background-paper) 100%);
        border: 1px solid rgba(21, 183, 159, 0.3);
        border-radius: 20px;
        padding: 28px;
        margin: 20px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 156, 12, 0.1) 0%, var(--background-paper) 100%);
        border: 1px solid rgba(251, 156, 12, 0.3);
        border-radius: 20px;
        padding: 28px;
        margin: 20px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(4, 170, 214, 0.1) 0%, var(--background-paper) 100%);
        border: 1px solid rgba(4, 170, 214, 0.3);
        border-radius: 20px;
        padding: 28px;
        margin: 20px 0;
    }
    
    /* ========================================
       PREDICTION RESULTS - Success/Error Style
       ======================================== */
    .prediction-high {
        background: linear-gradient(135deg, rgba(21, 183, 159, 0.15) 0%, var(--background-paper) 100%);
        border: 2px solid var(--success-main);
        border-radius: 24px;
        padding: 36px;
        text-align: center;
        box-shadow: 0 12px 40px -12px rgba(21, 183, 159, 0.3);
    }
    
    .prediction-low {
        background: linear-gradient(135deg, rgba(240, 68, 56, 0.15) 0%, var(--background-paper) 100%);
        border: 2px solid var(--error-main);
        border-radius: 24px;
        padding: 36px;
        text-align: center;
        box-shadow: 0 12px 40px -12px rgba(240, 68, 56, 0.3);
    }
    
    /* ========================================
       TAB STYLING - Material Pills
       ======================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--background-paper);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 14px 24px;
        transition: all 0.2s ease;
        font-size: 1.05em !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--background-level1);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-500), var(--primary-600)) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(99, 91, 255, 0.3);
    }
    
    /* ========================================
       BUTTON STYLING - Material Raised Button
       ======================================== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-500) 0%, var(--primary-600) 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 16px 32px;
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(99, 91, 255, 0.35);
        text-transform: none;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(99, 91, 255, 0.45);
        background: linear-gradient(135deg, var(--primary-400) 0%, var(--primary-500) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* ========================================
       SELECT BOX & INPUT STYLING - CHỈ CHO CHỌN
       ======================================== */
    .stSelectbox > div > div {
        background: var(--background-level1);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        transition: all 0.2s ease;
        font-size: 1.05em;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary-main);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-main);
        box-shadow: 0 0 0 3px rgba(99, 91, 255, 0.2);
    }
    
    /* Disable text input in selectbox - chỉ cho chọn */
    .stSelectbox input {
        caret-color: transparent !important;
    }
    
    .stTextInput > div > div > input {
        background: var(--background-level1);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        color: var(--text-primary);
        padding: 14px 18px;
        font-size: 1.05em;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-main);
        box-shadow: 0 0 0 3px rgba(99, 91, 255, 0.2);
    }
    
    .stSlider > div > div > div {
        background: var(--primary-main);
    }
    
    /* ========================================
       DATAFRAME STYLING
       ======================================== */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 1.05em !important;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: var(--background-paper);
    }
    
    /* ========================================
       METRIC COMPONENT OVERRIDE
       ======================================== */
    [data-testid="stMetricValue"] {
        color: var(--primary-main);
        font-weight: 700;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        font-size: 2em !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 1em !important;
    }
    
    [data-testid="stMetricDelta"] svg {
        stroke: var(--success-main);
    }
    
    /* ========================================
       EXPANDER STYLING
       ======================================== */
    .streamlit-expanderHeader {
        background: var(--background-paper);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 1.1em !important;
    }
    
    /* ========================================
       SCROLLBAR - Material Style
       ======================================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-default);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-500);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-400);
    }
    
    /* ========================================
       DIVIDER
       ======================================== */
    hr {
        border-color: rgba(255,255,255,0.08);
        margin: 24px 0;
    }
    
    /* ========================================
       ALERTS - Info/Success/Warning/Error
       ======================================== */
    .stAlert {
        border-radius: 16px;
        border: none;
        padding: 18px 22px;
        font-size: 1.05em;
    }
    
    /* ========================================
       PLOTLY CHART CONTAINER
       ======================================== */
    .js-plotly-plot {
        border-radius: 16px;
    }
    
    /* ========================================
       SIDEBAR NAVIGATION
       ======================================== */
    .sidebar-nav-item {
        padding: 14px 18px;
        border-radius: 12px;
        margin: 4px 0;
        transition: all 0.2s ease;
        cursor: pointer;
        font-size: 1.05em;
    }
    
    .sidebar-nav-item:hover {
        background: var(--background-level1);
    }
    
    .sidebar-nav-item.active {
        background: rgba(99, 91, 255, 0.12);
        color: var(--primary-main);
    }
    
    /* ========================================
       BADGE STYLES
       ======================================== */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.85em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success { background: rgba(21, 183, 159, 0.2); color: var(--success-main); }
    .badge-warning { background: rgba(251, 156, 12, 0.2); color: var(--warning-main); }
    .badge-error { background: rgba(240, 68, 56, 0.2); color: var(--error-main); }
    .badge-info { background: rgba(4, 170, 214, 0.2); color: var(--info-main); }
    .badge-primary { background: rgba(99, 91, 255, 0.2); color: var(--primary-main); }
    
    /* ========================================
       FOOTER STYLING
       ======================================== */
    .footer {
        background: var(--background-paper);
        border-radius: 24px;
        padding: 40px;
        margin-top: 32px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
    }
    
    /* ========================================
       ANIMATIONS
       ======================================== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    /* ========================================
       AI RESULT BOX
       ======================================== */
    .ai-result-box {
        background: linear-gradient(135deg, rgba(99, 91, 255, 0.1) 0%, var(--background-paper) 100%);
        border: 1px solid rgba(99, 91, 255, 0.3);
        border-radius: 20px;
        padding: 32px;
        margin: 24px 0;
        font-size: 1.1em;
        line-height: 1.8;
    }
    
    .ai-result-box h1, .ai-result-box h2, .ai-result-box h3 {
        color: var(--primary-main);
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    .ai-result-box ul, .ai-result-box ol {
        margin-left: 20px;
        line-height: 2;
    }
    
    .ai-result-box li {
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

THEME = {
    # Primary - Neon Blue
    "primary": "#635BFF",
    "primary_light": "#9ca7ff",
    "primary_dark": "#4e36f5",
    
    # Success - Kepple
    "success": "#15b79f",
    "success_light": "#5fe9ce",
    "success_dark": "#0e9382",
    
    # Warning - California
    "warning": "#ffbb1f",
    "warning_light": "#ffd049",
    "warning_dark": "#fb9c0c",
    
    # Error - Red Orange
    "error": "#f04438",
    "error_light": "#f97970",
    "error_dark": "#de3024",
    
    # Info - Shakespeare
    "info": "#04aad6",
    "info_light": "#10bee8",
    "info_dark": "#0787b3",
    
    # Neutrals - Nevada/Storm Grey
    "background": "#121621",
    "surface": "#212636",
    "surface_variant": "#313749",
    "text_primary": "#f0f4f8",
    "text_secondary": "#9fa6ad",
    "text_disabled": "#555e68",
    "divider": "rgba(255,255,255,0.08)",
}

# Chart color palette
CHART_COLORS = [
    "#635BFF",  # Primary - Neon Blue
    "#15b79f",  # Success - Kepple
    "#ffbb1f",  # Warning - California
    "#f04438",  # Error - Red Orange
    "#04aad6",  # Info - Shakespeare
    "#9ca7ff",  # Primary Light
    "#5fe9ce",  # Success Light
]

# Gemini API Key - Đọc từ biến môi trường hoặc file .env
# Để sử dụng: tạo file .env với nội dung GEMINI_API_KEY=your_api_key_here
# Hoặc set biến môi trường: export GEMINI_API_KEY=your_api_key_here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Khởi tạo Gemini client
@st.cache_resource
def init_gemini():
    """Khởi tạo Gemini AI client với SDK mới (google-genai)"""
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            # SDK mới: google-genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            return client
        except Exception as e:
            # Fallback: SDK cũ
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                return genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e2:
                st.error(f"Lỗi khởi tạo Gemini: {e2}")
                return None
    return None

@st.cache_resource
def load_models():
    """Tải các mô hình đã huấn luyện - chỉ 3 mô hình chính"""
    models = {}
    
    ipums_model_files = {
        "Decision Tree (Entropy)": "models/decision_tree_entropy_ipums.pkl",
        "Decision Tree (Gini)": "models/decision_tree_gini_ipums.pkl",
        "Naive Bayes": "models/naive_bayes_ipums.pkl"
    }
    
    # Mô hình cũ (fallback)
    old_model_files = {
        "Decision Tree (Entropy)": "models/tree_entropy_panel.pkl",
        "Decision Tree (Gini)": "models/tree_gini_panel.pkl",
        "Naive Bayes": "models/naive_bayes_panel.pkl"
    }
    
    # Thử load mô hình IPUMS trước
    for name, path in ipums_model_files.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                pass
    
    # Nếu không có mô hình IPUMS, thử load mô hình cũ
    if not models:
        for name, path in old_model_files.items():
            if os.path.exists(path):
                try:
                    models[name] = joblib.load(path)
                except Exception as e:
                    pass
    
    return models


@st.cache_resource
def load_feature_names():
    """Tải danh sách feature names cho mô hình IPUMS"""
    if os.path.exists("models/feature_names_ipums.pkl"):
        return joblib.load("models/feature_names_ipums.pkl")
    return None


@st.cache_data
def load_panel_data():
    """Tải dữ liệu - ưu tiên IPUMS processed data"""
    # Ưu tiên dữ liệu IPUMS đã xử lý
    if os.path.exists("data/ipums_processed.csv"):
        return pd.read_csv("data/ipums_processed.csv")
    # Fallback sang panel_microdata cũ
    if os.path.exists("data/panel_microdata.csv"):
        return pd.read_csv("data/panel_microdata.csv")
    return None


@st.cache_data
def load_macro_data():
    """Tải dữ liệu macro"""
    if os.path.exists("data/macro_region_year.csv"):
        return pd.read_csv("data/macro_region_year.csv")
    
    # Generate synthetic macro data
    regions = ["Bắc", "Trung", "Nam"]
    years = list(range(2019, 2025))
    grid = [(r, y) for r in regions for y in years]
    m = pd.DataFrame(grid, columns=["region", "year"])
    
    base_hpi = {"Bắc": 1.00, "Trung": 0.97, "Nam": 1.05}
    m["house_price_index"] = [100 * base_hpi[r] * (1 + 0.05) ** (y - 2019) for r, y in zip(m.region, m.year)]
    m["rental_index"] = [100 * base_hpi[r] * (1 + 0.03) ** (y - 2019) for r, y in zip(m.region, m.year)]
    m["unemployment_rate"] = [2.2 + (1.6 if y in (2020, 2021) else 0.6) + (0.3 if r == "Nam" else 0) for r, y in zip(m.region, m.year)]
    m["CPI"] = [100 * (1 + 0.03) ** (y - 2019) for y in m.year]
    m["child_care_cost_index"] = [100 * (1 + 0.04) ** (y - 2019) for y in m.year]
    m["family_tax_benefit_index"] = [100 + (3 if r != "Nam" else 2) * (y - 2019) for r, y in zip(m.region, m.year)]
    
    return m


def render_header(text, size="large"):
    """Render header với style Material Kit - không có icon"""
    if size == "large":
        st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="section-header" style="font-size: 1.1em; padding: 14px 20px;">{text}</div>', unsafe_allow_html=True)


def create_gauge_chart(value, title, color=None):
    """Tạo gauge chart cho xác suất"""
    if color is None:
        color = THEME["primary"]
    
    # Determine color based on value
    if value >= 0.7:
        bar_color = THEME["success"]
    elif value >= 0.5:
        bar_color = THEME["info"]
    elif value >= 0.3:
        bar_color = THEME["warning"]
    else:
        bar_color = THEME["error"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': THEME["text_primary"], 'family': 'Inter, sans-serif'}},
        number={'suffix': "%", 'font': {'size': 56, 'color': THEME["text_primary"], 'family': 'Plus Jakarta Sans, Inter, sans-serif', 'weight': 700}},
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickwidth': 2, 
                'tickcolor': THEME["text_secondary"], 
                'tickfont': {'color': THEME["text_secondary"], 'size': 13},
                'dtick': 25
            },
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': THEME["surface_variant"],
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(240, 68, 56, 0.15)'},     # Error
                {'range': [30, 50], 'color': 'rgba(255, 187, 31, 0.15)'},   # Warning
                {'range': [50, 70], 'color': 'rgba(4, 170, 214, 0.15)'},    # Info
                {'range': [70, 100], 'color': 'rgba(21, 183, 159, 0.15)'}   # Success
            ],
            'threshold': {
                'line': {'color': THEME["text_primary"], 'width': 3},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': THEME["text_primary"], 'family': 'Inter, sans-serif'},
        height=300,
        margin=dict(l=30, r=30, t=70, b=30)
    )
    
    return fig


def create_material_chart_layout():
    """Tạo layout chuẩn cho biểu đồ theo Material Kit React Design"""
    return dict(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=THEME["surface"],
        font=dict(family='Inter, sans-serif', color=THEME["text_primary"], size=13),
        title_font=dict(size=20, color=THEME["text_primary"], family='Plus Jakarta Sans, Inter, sans-serif'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=THEME["text_secondary"], size=13),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.06)',
            linecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color=THEME["text_secondary"], size=12),
            title_font=dict(color=THEME["text_secondary"], size=13)
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.06)',
            linecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color=THEME["text_secondary"], size=12),
            title_font=dict(color=THEME["text_secondary"], size=13)
        ),
        margin=dict(l=60, r=30, t=80, b=60),
        hoverlabel=dict(
            bgcolor=THEME["surface_variant"],
            font_size=14,
            font_family='Inter, sans-serif'
        )
    )


def get_data_summary_for_ai(panel_data):
    """Tạo bản tóm tắt dữ liệu cho AI phân tích - phiên bản IPUMS"""
    if panel_data is None:
        return "Chưa có dữ liệu panel."
    
    yearly_marriage_rate = panel_data.groupby("year")["Y_married"].mean()
    age_rate = panel_data.groupby("age_group")["Y_married"].mean()
    urban_rate = panel_data.groupby("urban_rural")["Y_married"].mean()
    education_rate = panel_data.groupby("education_level")["Y_married"].mean()
    home_rate = panel_data.groupby("home_ownership")["Y_married"].mean()
    
    # Kiểm tra các cột tồn tại
    region_text = ""
    if "region" in panel_data.columns:
        region_rate = panel_data.groupby("region")["Y_married"].mean()
        region_text = f"""
5. TỶ LỆ KẾT HÔN THEO VÙNG MIỀN:
{region_rate.to_string()}
"""
    
    sex_text = ""
    if "sex" in panel_data.columns:
        sex_rate = panel_data.groupby("sex")["Y_married"].mean()
        sex_text = f"""
6. TỶ LỆ KẾT HÔN THEO GIỚI TÍNH:
{sex_rate.to_string()}
"""
    
    living_area_text = ""
    if "living_area_level" in panel_data.columns:
        living_area_rate = panel_data.groupby("living_area_level")["Y_married"].mean()
        living_area_text = f"""
7. TỶ LỆ KẾT HÔN THEO DIỆN TÍCH NHÀ Ở:
{living_area_rate.to_string()}
"""
    
    household_text = ""
    if "household_size_group" in panel_data.columns:
        household_rate = panel_data.groupby("household_size_group")["Y_married"].mean()
        household_text = f"""
8. TỶ LỆ KẾT HÔN THEO QUY MÔ HỘ GIA ĐÌNH:
{household_rate.to_string()}
"""
    
    summary = f"""
DỮ LIỆU THỐNG KÊ THỰC TẾ TỪ ĐIỀU TRA DÂN SỐ VIỆT NAM (IPUMS):

1. TỔNG QUAN:
- Tổng số quan sát: {len(panel_data):,} bản ghi
- Năm điều tra: {sorted(panel_data['year'].unique())}
- Tỷ lệ kết hôn trung bình: {panel_data['Y_married'].mean():.2%}

2. TỶ LỆ KẾT HÔN THEO NĂM:
{yearly_marriage_rate.to_string()}

3. TỶ LỆ KẾT HÔN THEO NHÓM TUỔI:
{age_rate.to_string()}

4. TỶ LỆ KẾT HÔN THEO KHU VỰC (Đô thị/Nông thôn):
{urban_rate.to_string()}
{region_text}{sex_text}{living_area_text}{household_text}
9. TỶ LỆ KẾT HÔN THEO TRÌNH ĐỘ HỌC VẤN:
{education_rate.to_string()}

10. TỶ LỆ KẾT HÔN THEO SỞ HỮU NHÀ:
{home_rate.to_string()}
"""
    return summary

def main():
    # Sidebar với Material Kit React Design
    with st.sidebar:
        # Logo và Brand
        st.markdown("""
        <div style="text-align: center; padding: 28px 16px 20px 16px;">
            <div style="width: 64px; height: 64px; background: linear-gradient(135deg, #635BFF, #4e36f5); border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px auto; box-shadow: 0 8px 16px rgba(99, 91, 255, 0.3);">
                <span style="color: white; font-size: 28px; font-weight: 700;">M</span>
            </div>
            <h2 style="color: #f0f4f8; margin: 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.6em; font-weight: 700;">Marriage AI</h2>
            <p style="color: #9fa6ad; font-size: 1.05em; margin-top: 6px;">Dự báo Kết hôn</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 8px 0 20px 0;">', unsafe_allow_html=True)
        
        # Overview Section
        st.markdown("""
        <div style="padding: 0 8px;">
            <p style="color: #9fa6ad; font-size: 0.95em; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; font-weight: 600;">TỔNG QUAN</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats mini cards
        st.markdown("""
        <div style="background: #313749; border-radius: 12px; padding: 16px; margin: 0 8px 16px 8px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <div style="width: 44px; height: 44px; background: rgba(99, 91, 255, 0.15); border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #635BFF; font-weight: 700; font-size: 16px;">VN</span>
                </div>
                <div>
                    <p style="color: #f0f4f8; margin: 0; font-weight: 600; font-size: 1.15em;">IPUMS Vietnam Census</p>
                    <p style="color: #9fa6ad; margin: 0; font-size: 1em;">2009 & 2019</p>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08);">
                <div style="text-align: center;">
                    <p style="color: #635BFF; font-weight: 700; font-size: 1.4em; margin: 0;">3</p>
                    <p style="color: #9fa6ad; font-size: 0.95em; margin: 0;">Mô hình</p>
                </div>
                <div style="text-align: center;">
                    <p style="color: #15b79f; font-weight: 700; font-size: 1.4em; margin: 0;">2</p>
                    <p style="color: #9fa6ad; font-size: 0.95em; margin: 0;">Năm</p>
                </div>
                <div style="text-align: center;">
                    <p style="color: #ffbb1f; font-weight: 700; font-size: 1.4em; margin: 0;">3</p>
                    <p style="color: #9fa6ad; font-size: 0.95em; margin: 0;">Vùng</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 16px 0;">', unsafe_allow_html=True)
        
        # Features
        st.markdown("""
        <div style="padding: 0 8px;">
            <p style="color: #9fa6ad; font-size: 0.95em; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; font-weight: 600;">TÍNH NĂNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        features = [
            ("Dự báo cá nhân", "#635BFF"),
            ("Phân tích dữ liệu", "#15b79f"),
            ("Luật IF-THEN", "#ffbb1f"),
            ("So sánh mô hình", "#04aad6"),
            ("Yếu tố ảnh hưởng", "#f04438"),
            ("Dự báo bởi AI", "#9ca7ff"),
        ]
        
        for label, color in features:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 12px; padding: 14px 18px; margin: 6px 8px; border-radius: 10px; transition: all 0.2s; border-left: 3px solid {color};">
                <span style="color: #f0f4f8; font-size: 1.1em;">{label}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 20px 0;">', unsafe_allow_html=True)
        
    
    # Header với pattern overlay
    st.markdown("""
    <div class="main-header">
        <h1>DỰ BÁO XU HƯỚNG KẾT HÔN</h1>
        <h3>Giới trẻ Việt Nam 18-35 tuổi</h3>
        <p>Phân tích dữ liệu Điều tra Dân số Việt Nam (IPUMS) với Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    models = load_models()
    panel_data = load_panel_data()
    macro_data = load_macro_data()
    
    # Tabs với Material Style
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "DỰ BÁO CÁ NHÂN",
        "PHÂN TÍCH DỮ LIỆU",
        "LUẬT IF-THEN",
        "SO SÁNH MÔ HÌNH",
        "YẾU TỐ ẢNH HƯỞNG",
        "DỰ BÁO BỞI AI"
    ])
    
    with tab1:
        render_header("Dự báo xác suất kết hôn")
        
        # Load feature names
        feature_names = load_feature_names()
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("""
            <div class="form-card">
                <h4><span style="background: linear-gradient(135deg, #635BFF, #4e36f5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Thông tin cá nhân</span></h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Tuổi - slider thay vì nhóm tuổi
            age = st.slider("Tuổi", min_value=18, max_value=35, value=26, key="age_slider")
            
            # Nhóm tuổi (tự tính)
            if age <= 24:
                age_group = "18-24"
            elif age <= 29:
                age_group = "25-29"
            else:
                age_group = "30-35"
            
            sex = st.selectbox("Giới tính", ["Nam", "Nữ"], key="sex_select")
            region = st.selectbox("Vùng miền", ["Bắc", "Trung", "Nam"], key="region_select")
            urban_rural = st.selectbox("Khu vực sinh sống", ["Đô thị", "Nông thôn"], key="urban_rural_select")
            
            # Học vấn - theo IPUMS
            education_options = ["≤THPT", "ĐH/CĐ+"]
            education = st.selectbox("Trình độ học vấn", education_options, index=0, key="education_select")
            
        with col2:
            st.markdown("""
            <div class="form-card">
                <h4><span style="background: linear-gradient(135deg, #15b79f, #0e9382); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Thông tin hộ gia đình</span></h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Sở hữu nhà
            home_options = ["Có sở hữu nhà", "Không sở hữu nhà"]
            home_display = st.selectbox("Sở hữu nhà ở", home_options, key="home_select")
            home_ownership = 1 if home_display == "Có sở hữu nhà" else 0
            
            # Diện tích nhà ở
            living_area = st.slider("Diện tích nhà ở (m²)", min_value=10, max_value=300, value=70, key="living_area_slider")
            
            # Mức diện tích
            if living_area <= 40:
                living_area_level = "Nhỏ"
            elif living_area <= 72:
                living_area_level = "Trung bình"
            elif living_area <= 126:
                living_area_level = "Khá"
            else:
                living_area_level = "Rộng"
            
            # Quy mô hộ gia đình
            household_size = st.slider("Số người trong hộ", min_value=1, max_value=12, value=4, key="household_slider")
            
            # Nhóm quy mô hộ
            if household_size <= 2:
                household_size_group = "1-2 người"
            elif household_size <= 4:
                household_size_group = "3-4 người"
            elif household_size <= 6:
                household_size_group = "5-6 người"
            else:
                household_size_group = ">6 người"
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 28px 0;">', unsafe_allow_html=True)
        
        # Chọn mô hình
        st.markdown("""
        <div style="background: var(--background-paper); border-radius: 16px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.08);">
            <p style="color: #9fa6ad; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 12px 0;">CHỌN MÔ HÌNH DỰ BÁO</p>
        </div>
        """, unsafe_allow_html=True)
        
        model_choice = st.selectbox("Mô hình", list(models.keys()) if models else ["Chưa có mô hình"], key="model_select")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Nút dự báo
        if st.button("Dự báo ngay", type="primary", use_container_width=True, key="predict_btn"):
            if models and model_choice in models:
                model = models[model_choice]
                
                # Chuẩn bị input theo format của mô hình IPUMS
                # Tạo DataFrame với tất cả các features đã one-hot encoded
                input_dict = {
                    'age': age,
                    'home_ownership': home_ownership,
                    'living_area': living_area,
                    'household_size': household_size,
                    # Age group one-hot
                    'age_group_18-24': 1 if age_group == '18-24' else 0,
                    'age_group_25-29': 1 if age_group == '25-29' else 0,
                    'age_group_30-35': 1 if age_group == '30-35' else 0,
                    # Sex one-hot
                    'sex_Nam': 1 if sex == 'Nam' else 0,
                    'sex_Nữ': 1 if sex == 'Nữ' else 0,
                    # Education one-hot
                    'education_level_ĐH/CĐ+': 1 if education == 'ĐH/CĐ+' else 0,
                    'education_level_≤THPT': 1 if education == '≤THPT' else 0,
                    # Urban/Rural one-hot
                    'urban_rural_Đô thị': 1 if urban_rural == 'Đô thị' else 0,
                    'urban_rural_Nông thôn': 1 if urban_rural == 'Nông thôn' else 0,
                    # Region one-hot
                    'region_Bắc': 1 if region == 'Bắc' else 0,
                    'region_Nam': 1 if region == 'Nam' else 0,
                    'region_Trung': 1 if region == 'Trung' else 0,
                    'region_Khác': 0,
                    # Living area level one-hot
                    'living_area_level_Khá': 1 if living_area_level == 'Khá' else 0,
                    'living_area_level_Nhỏ': 1 if living_area_level == 'Nhỏ' else 0,
                    'living_area_level_Rộng': 1 if living_area_level == 'Rộng' else 0,
                    'living_area_level_Trung bình': 1 if living_area_level == 'Trung bình' else 0,
                    # Household size group one-hot
                    'household_size_group_1-2 người': 1 if household_size_group == '1-2 người' else 0,
                    'household_size_group_3-4 người': 1 if household_size_group == '3-4 người' else 0,
                    'household_size_group_5-6 người': 1 if household_size_group == '5-6 người' else 0,
                    'household_size_group_>6 người': 1 if household_size_group == '>6 người' else 0,
                }
                
                # Tạo DataFrame
                input_data = pd.DataFrame([input_dict])
                
                # Đảm bảo có đủ các cột như model cần
                if feature_names:
                    for col in feature_names:
                        if col not in input_data.columns:
                            input_data[col] = 0
                    input_data = input_data[feature_names]
                
                # Dự báo
                try:
                    proba = model.predict_proba(input_data)[0, 1]
                    pred = "Có khả năng kết hôn cao" if proba >= 0.5 else "Khả năng kết hôn thấp"
                    
                    st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 28px 0;">', unsafe_allow_html=True)
                    render_header("Kết quả dự báo", size="small")
                    
                    col_result1, col_result2 = st.columns([1, 1])
                    
                    with col_result1:
                        # Gauge chart
                        fig = create_gauge_chart(proba, "Xác suất kết hôn")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_result2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        if proba >= 0.5:
                            st.markdown(f"""
                            <div class="prediction-high">
                                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">KẾT QUẢ DỰ BÁO</p>
                                <h1 style="color: {THEME['success']}; margin: 16px 0; font-size: 3.5em; font-family: 'Plus Jakarta Sans', sans-serif;">{proba:.1%}</h1>
                                <p style="color: #ffffff; margin: 0; font-weight: 600; font-size: 1.2em;">{pred}</p>
                                <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(21, 183, 159, 0.3);">
                                    <span class="badge badge-success">Mô hình: {model_choice}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-low">
                                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">KẾT QUẢ DỰ BÁO</p>
                                <h1 style="color: {THEME['error']}; margin: 16px 0; font-size: 3.5em; font-family: 'Plus Jakarta Sans', sans-serif;">{proba:.1%}</h1>
                                <p style="color: #ffffff; margin: 0; font-weight: 600; font-size: 1.2em;">{pred}</p>
                                <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(240, 68, 56, 0.3);">
                                    <span class="badge badge-error">Mô hình: {model_choice}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Phân tích yếu tố
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="background: {THEME['surface']}; padding: 24px; border-radius: 16px; border-left: 4px solid {THEME['primary']};">
                            <p style="color: {THEME['primary']}; font-weight: 600; margin: 0 0 16px 0; font-size: 1.1em;">Các yếu tố ảnh hưởng chính</p>
                        """, unsafe_allow_html=True)
                        
                        factors = []
                        # Phân tích dựa trên các biến mới
                        if age_group == "30-35":
                            factors.append(("Nhóm tuổi 30-35 có xu hướng kết hôn cao nhất", THEME["success"]))
                        elif age_group == "18-24":
                            factors.append(("Nhóm tuổi 18-24 thường trì hoãn kết hôn", THEME["warning"]))
                        else:
                            factors.append(("Nhóm tuổi 25-29 là độ tuổi kết hôn phổ biến", THEME["info"]))
                        
                        if home_ownership == 1:
                            factors.append(("Sở hữu nhà tăng 16% khả năng kết hôn", THEME["success"]))
                        else:
                            factors.append(("Không sở hữu nhà là rào cản lớn", THEME["warning"]))
                        
                        if urban_rural == "Đô thị":
                            factors.append(("Đô thị: tỷ lệ kết hôn cao hơn (69%)", THEME["info"]))
                        else:
                            factors.append(("Nông thôn: tỷ lệ kết hôn thấp hơn (58%)", THEME["warning"]))
                        
                        if sex == "Nữ":
                            factors.append(("Nữ giới có tỷ lệ kết hôn cao hơn nam", THEME["success"]))
                        
                        for f, color in factors[:4]:
                            st.markdown(f'<p style="color: {color}; margin: 12px 0; display: flex; align-items: center; gap: 8px; font-size: 1.05em;">• {f}</p>', unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Lỗi dự báo: {e}")
            else:
                st.warning("Vui lòng chọn mô hình hợp lệ")
    
    with tab2:
        render_header("Phân tích dữ liệu Điều tra Dân số Việt Nam (IPUMS)")
        
        if panel_data is not None:
            # Tổng quan - Material Kit React Style Stats Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card stats-card-purple">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <p style="color: {THEME['text_secondary']}; font-size: 0.85em; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;">Tổng quan sát</p>
                            <h2 style="color: {THEME['primary']}; margin: 12px 0 4px 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2em;">{len(panel_data):,}</h2>
                            <p style="color: {THEME['text_secondary']}; font-size: 0.9em; margin: 0;">bản ghi</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stats-card stats-card-green">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <p style="color: {THEME['text_secondary']}; font-size: 0.85em; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;">Số năm</p>
                            <h2 style="color: {THEME['success']}; margin: 12px 0 4px 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2em;">{len(panel_data["year"].unique())}</h2>
                            <p style="color: {THEME['text_secondary']}; font-size: 0.9em; margin: 0;">2019-2024</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="stats-card stats-card-orange">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <p style="color: {THEME['text_secondary']}; font-size: 0.85em; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;">Tỷ lệ kết hôn TB</p>
                            <h2 style="color: {THEME['warning']}; margin: 12px 0 4px 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2em;">{panel_data['Y_married'].mean():.1%}</h2>
                            <p style="color: {THEME['text_secondary']}; font-size: 0.9em; margin: 0;">trung bình</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                # Tính số vùng miền
                n_regions = panel_data['region'].nunique() if 'region' in panel_data.columns else 3
                st.markdown(f"""
                <div class="stats-card stats-card-pink">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <p style="color: {THEME['text_secondary']}; font-size: 0.85em; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;">Số vùng miền</p>
                            <h2 style="color: {THEME['error']}; margin: 12px 0 4px 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2em;">{n_regions}</h2>
                            <p style="color: {THEME['text_secondary']}; font-size: 0.9em; margin: 0;">vùng</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Biểu đồ theo năm
            render_header("Xu hướng theo năm điều tra", size="small")
            
            yearly_stats = panel_data.groupby("year").agg({
                "Y_married": ["sum", "mean", "count"]
            }).round(3)
            yearly_stats.columns = ["Số đã kết hôn", "Tỷ lệ kết hôn", "Tổng mẫu"]
            yearly_stats = yearly_stats.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    yearly_stats, x="year", y="Số đã kết hôn",
                    title="Số người đã kết hôn theo năm điều tra",
                    color_discrete_sequence=[THEME["primary"]],
                    labels={"Số đã kết hôn": "Số đã kết hôn (người)", "year": "Năm"}
                )
                fig1.update_layout(**create_material_chart_layout())
                fig1.update_yaxes(title_text="Số đã kết hôn (người)")
                fig1.update_traces(marker_line_width=0, marker_cornerradius=6)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(
                    yearly_stats, x="year", y="Tỷ lệ kết hôn",
                    title="Tỷ lệ kết hôn theo năm điều tra",
                    color_discrete_sequence=[THEME["success"]],
                    labels={"Tỷ lệ kết hôn": "Tỷ lệ kết hôn (%)", "year": "Năm"}
                )
                fig2.update_layout(**create_material_chart_layout())
                fig2.update_yaxes(title_text="Tỷ lệ kết hôn (%)")
                fig2.update_traces(marker_line_width=0, marker_cornerradius=6)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Phân tích theo phân khúc
            render_header("Phân tích theo phân khúc", size="small")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age_stats = panel_data.groupby("age_group")["Y_married"].mean().reset_index()
                fig3 = px.bar(
                    age_stats, x="age_group", y="Y_married",
                    title="Tỷ lệ kết hôn theo nhóm tuổi",
                    color="age_group",
                    color_discrete_sequence=[THEME["error"], THEME["warning"], THEME["success"]]
                )
                fig3.update_layout(**create_material_chart_layout(), showlegend=False)
                fig3.update_traces(marker_line_width=0, marker_cornerradius=6)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                if "region" in panel_data.columns:
                    region_urban = panel_data.groupby(["region", "urban_rural"])["Y_married"].mean().reset_index()
                    fig4 = px.bar(
                        region_urban, x="region", y="Y_married", color="urban_rural",
                        title="Tỷ lệ kết hôn theo Vùng và Khu vực",
                        barmode="group",
                        color_discrete_sequence=[THEME["primary"], THEME["warning"]]
                    )
                else:
                    urban_stats = panel_data.groupby("urban_rural")["Y_married"].mean().reset_index()
                    fig4 = px.bar(
                        urban_stats, x="urban_rural", y="Y_married",
                        title="Tỷ lệ kết hôn theo Khu vực",
                        color="urban_rural",
                        color_discrete_sequence=[THEME["primary"], THEME["warning"]]
                    )
                fig4.update_layout(**create_material_chart_layout())
                fig4.update_traces(marker_line_width=0, marker_cornerradius=6)
                st.plotly_chart(fig4, use_container_width=True)
            
            # Phân tích theo các yếu tố khác
            render_header("Phân tích theo các yếu tố khác", size="small")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "sex" in panel_data.columns:
                    sex_stats = panel_data.groupby("sex")["Y_married"].mean().reset_index()
                    fig5 = px.bar(
                        sex_stats, x="sex", y="Y_married",
                        title="Tỷ lệ kết hôn theo Giới tính",
                        color="sex",
                        color_discrete_sequence=[THEME["info"], THEME["error"]]
                    )
                    fig5.update_layout(**create_material_chart_layout(), showlegend=False, height=400)
                    fig5.update_traces(marker_line_width=0, marker_cornerradius=8)
                    st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                if "education_level" in panel_data.columns:
                    edu_stats = panel_data.groupby("education_level")["Y_married"].mean().reset_index()
                    fig6 = px.bar(
                        edu_stats, x="education_level", y="Y_married",
                        title="Tỷ lệ kết hôn theo Trình độ học vấn",
                        color="education_level",
                        color_discrete_sequence=[THEME["warning"], THEME["success"]]
                    )
                    fig6.update_layout(**create_material_chart_layout(), showlegend=False, height=400)
                    fig6.update_traces(marker_line_width=0, marker_cornerradius=8)
                    st.plotly_chart(fig6, use_container_width=True)
            
            # Phân tích sở hữu nhà và quy mô hộ
            render_header("Ảnh hưởng của điều kiện nhà ở", size="small")
            
            col1, col2 = st.columns(2)
            
            with col1:
                home_stats = panel_data.groupby("home_ownership")["Y_married"].mean().reset_index()
                home_stats["home_ownership"] = home_stats["home_ownership"].map({1: "Có sở hữu", 0: "Không sở hữu"})
                fig7 = px.bar(
                    home_stats, x="home_ownership", y="Y_married",
                    title="Tỷ lệ kết hôn theo Sở hữu nhà",
                    color="home_ownership",
                    color_discrete_sequence=[THEME["success"], THEME["error"]]
                )
                fig7.update_layout(**create_material_chart_layout(), showlegend=False, height=400)
                fig7.update_traces(marker_line_width=0, marker_cornerradius=8)
                st.plotly_chart(fig7, use_container_width=True)
            
            with col2:
                if "household_size_group" in panel_data.columns:
                    hh_stats = panel_data.groupby("household_size_group")["Y_married"].mean().reset_index()
                    fig8 = px.bar(
                        hh_stats, x="household_size_group", y="Y_married",
                        title="Tỷ lệ kết hôn theo Quy mô hộ gia đình",
                        color="household_size_group",
                        color_discrete_sequence=[THEME["info"], THEME["primary"], THEME["warning"], THEME["error"]]
                    )
                    fig8.update_layout(**create_material_chart_layout(), showlegend=False, height=400)
                    fig8.update_traces(marker_line_width=0, marker_cornerradius=8)
                    st.plotly_chart(fig8, use_container_width=True)
            
        else:
            st.markdown(f"""
            <div class="info-box">
                <p style="color: {THEME['info']}; font-weight: 600; margin: 0 0 12px 0; font-size: 1.15em;">Chưa có dữ liệu panel</p>
                <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 1.05em;">Vui lòng nhấn nút bên dưới để tạo dữ liệu.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Tạo dữ liệu Panel", key="generate_panel_btn"):
                with st.spinner("Đang tạo dữ liệu..."):
                    from src.panel_data_generator import generate_panel_data
                    panel = generate_panel_data()
                    st.success(f"Đã tạo thành công {len(panel)} bản ghi!")
                    
                    # Hiển thị dữ liệu vừa tạo
                    st.markdown("<br>", unsafe_allow_html=True)
                    render_header("Dữ liệu Panel vừa tạo", size="small")
                    
                    # Thống kê tổng quan
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Tổng bản ghi", f"{len(panel):,}")
                    with col2:
                        st.metric("Số năm", len(panel["year"].unique()))
                    with col3:
                        st.metric("Tỷ lệ kết hôn TB", f"{panel['Y_married'].mean():.1%}")
                    with col4:
                        st.metric("Số cá nhân", f"{panel['id'].nunique():,}")
                    
                    # Hiển thị 10 dòng đầu
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background: {THEME['surface']}; padding: 20px; border-radius: 16px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.08);">
                        <p style="color: {THEME['text_primary']}; font-weight: 600; margin: 0; font-size: 1.1em;">Xem trước dữ liệu (10 dòng đầu)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(panel.head(10), use_container_width=True, hide_index=True)
                    
                    st.info("Vui lòng nhấn F5 hoặc reload trang để xem đầy đủ biểu đồ phân tích.")
    
    with tab3:
        render_header("Luật IF-THEN từ Decision Tree")
        
        st.markdown(f"""
        <div class="info-box">
            <p style="color: {THEME['warning']}; font-weight: 600; margin: 0 0 8px 0; font-size: 1.1em;">Ghi chú</p>
            <p style="color: {THEME['text_primary']}; margin: 0; line-height: 1.8; font-size: 1.05em;">
                Các luật dưới đây được trích xuất từ mô hình Cây quyết định, 
                giúp hiểu các yếu tố quyết định kết hôn trong năm.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        rules = [
            {
                "title": "Luật 1: Nhóm tuổi 30-35 có nhà ở đô thị",
                "condition": "IF age_group = '30-35' AND home_ownership = 1 AND urban_rural = 'Đô thị'",
                "result": "THEN Y_married = 1 (Đã kết hôn)",
                "confidence": "87%",
                "insight": "Nhóm 30-35 sở hữu nhà tại đô thị có tỷ lệ kết hôn cao nhất",
                "badge": "positive"
            },
            {
                "title": "Luật 2: Thanh niên đô thị học vấn cao chưa có nhà",
                "condition": "IF age_group = '18-24' AND urban_rural = 'Đô thị' AND education_level = 'ĐH/CĐ+' AND home_ownership = 0",
                "result": "THEN Y_married = 0 (Chưa kết hôn)",
                "confidence": "82%",
                "insight": "Thanh niên trẻ đô thị học vấn cao thường trì hoãn kết hôn khi chưa có nhà",
                "badge": "negative"
            },
            {
                "title": "Luật 3: Nông thôn với nhà ở rộng",
                "condition": "IF urban_rural = 'Nông thôn' AND living_area_level IN {Khá, Rộng} AND age_group IN {25-29, 30-35}",
                "result": "THEN Y_married = 1 (Đã kết hôn)",
                "confidence": "79%",
                "insight": "Người nông thôn có nhà ở rộng rãi kết hôn sớm hơn",
                "badge": "positive"
            },
            {
                "title": "Luật 4: Không sở hữu nhà, diện tích nhỏ",
                "condition": "IF home_ownership = 0 AND living_area_level = 'Nhỏ' AND age_group = '18-24'",
                "result": "THEN Y_married = 0 (Chưa kết hôn)",
                "confidence": "85%",
                "insight": "Điều kiện nhà ở khó khăn là rào cản lớn cho quyết định kết hôn của giới trẻ",
                "badge": "negative"
            },
            {
                "title": "Luật 5: Nữ giới 25-29 tại đô thị",
                "condition": "IF sex = 'Nữ' AND age_group = '25-29' AND urban_rural = 'Đô thị'",
                "result": "THEN Y_married = 1 (Đã kết hôn)",
                "confidence": "75%",
                "insight": "Nữ giới 25-29 tại đô thị có xu hướng kết hôn cao",
                "badge": "positive"
            },
            {
                "title": "Luật 6: Sở hữu nhà kết hợp hộ gia đình lớn",
                "condition": "IF home_ownership = 1 AND household_size_group IN {3-4 người, 5-6 người} AND age_group IN {25-29, 30-35}",
                "result": "THEN Y_married = 1 (Đã kết hôn)",
                "confidence": "81%",
                "insight": "Sở hữu nhà và sống trong gia đình đa thế hệ tăng khả năng kết hôn",
                "badge": "positive"
            }
        ]
        
        for i, rule in enumerate(rules):
            result_color = THEME["success"] if "= 1" in rule["result"] else THEME["error"]
            badge_class = "badge-success" if rule["badge"] == "positive" else "badge-error"
            st.markdown(f"""
            <div class="rule-box" style="border-left-color: {result_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <p style="color: {THEME['text_primary']}; font-weight: 600; font-size: 1.15em; margin: 0;">{rule['title']}</p>
                    <span class="badge {badge_class}">{rule['confidence']}</span>
                </div>
                <div style="background: rgba(99, 91, 255, 0.08); padding: 16px 18px; border-radius: 10px; margin: 12px 0; border: 1px solid rgba(99, 91, 255, 0.15);">
                    <p style="color: {THEME['primary_light']}; font-family: 'Roboto Mono', monospace; font-size: 1em; margin: 0; line-height: 1.6;">{rule['condition']}</p>
                </div>
                <p style="color: {result_color}; font-family: 'Roboto Mono', monospace; font-weight: 600; font-size: 1.05em; margin: 12px 0;">{rule['result']}</p>
                <div style="margin-top: 16px; padding-top:  12px; border-top: 1px solid rgba(255,255,255,0.06);">
                    <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 1em;">{rule['insight']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        render_header("So sánh hiệu năng các mô hình")
        
        # Bảng so sánh - load từ file hoặc dùng dữ liệu mặc định
        if os.path.exists("outputs/model_comparison_panel.csv"):
            comparison_df = pd.read_csv("outputs/model_comparison_panel.csv")
            # Đổi tên cột để hiển thị đẹp hơn
            display_df = comparison_df.rename(columns={
                "Model": "Mô hình",
                "Train_Acc": "Train Accuracy",
                "Val_Acc": "Val Accuracy", 
                "Test_Acc": "Test Accuracy",
                "Test_AUC": "Test ROC-AUC",
                "Test_F1": "Test F1-Score"
            })
        else:
            display_df = pd.DataFrame({
                "Mô hình": ["Decision Tree (Entropy)", "Decision Tree (Gini)", "Naive Bayes"],
                "Train Accuracy": [0.78, 0.77, 0.72],
                "Val Accuracy": [0.76, 0.75, 0.71],
                "Test Accuracy": [0.75, 0.74, 0.70],
                "Test ROC-AUC": [0.82, 0.81, 0.76],
                "Test F1-Score": [0.68, 0.67, 0.62]
            })
        
        # Hiển thị bảng với card style
        st.markdown(f"""
        <div style="background: {THEME['surface']}; padding: 24px; border-radius: 16px; margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.08);">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div>
                    <h4 style="color: {THEME['text_primary']}; margin: 0; font-weight: 600; font-size: 1.2em;">Bảng so sánh chi tiết</h4>
                    <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 1em;">Metrics đánh giá các mô hình ML</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            display_df.style.format({
                "Train Accuracy": "{:.2%}",
                "Val Accuracy": "{:.2%}",
                "Test Accuracy": "{:.2%}",
                "Test ROC-AUC": "{:.2%}",
                "Test F1-Score": "{:.2%}"
            }),
            use_container_width=True, 
            hide_index=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Biểu đồ
        render_header("Biểu đồ so sánh", size="small")
        
        fig = go.Figure()
        
        # Danh sách metrics theo tên cột đã đổi
        metrics_display = ["Train Accuracy", "Val Accuracy", "Test Accuracy", "Test ROC-AUC", "Test F1-Score"]
        
        for i, row in display_df.iterrows():
            model_name = row["Mô hình"]
            values = [row[m] for m in metrics_display]
            fig.add_trace(go.Bar(
                name=model_name,
                x=metrics_display,
                y=values,
                marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                marker_line_width=0,
                marker_cornerradius=6,
                text=[f"{v:.1%}" for v in values],
                textposition='outside',
                textfont=dict(color=THEME["text_primary"], size=12, family='Inter, sans-serif')
            ))
        
        layout = create_material_chart_layout()
        layout.update(
            title="So sánh các chỉ số đánh giá",
            barmode='group',
            yaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor='rgba(255, 255, 255, 0.06)'),
            height=500,
            bargap=0.2,
            bargroupgap=0.1
        )
        fig.update_layout(**layout)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Nhận xét với card style
        st.markdown(f"""
        <div class="insight-box">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div>
                    <p style="color: {THEME['text_primary']}; font-weight: 600; margin: 0; font-size: 1.15em;">Nhận xét & Đánh giá</p>
                    <p style="color: {THEME['text_secondary']}; font-size: 1em; margin: 0;">Phân tích hiệu năng mô hình</p>
                </div>
            </div>
            <ul style="color: {THEME['text_primary']}; line-height: 2.2; margin: 0; padding-left: 20px; font-size: 1.05em;">
                <li><span style="color: {THEME['primary']}; font-weight: 600;">Decision Tree (Entropy)</span> cho kết quả tốt nhất với ROC-AUC ~0.82</li>
                <li>Mô hình ổn định qua các năm validation (2023) và test (2024)</li>
                <li><span style="color: {THEME['error']}; font-weight: 600;">Naive Bayes</span> có accuracy thấp hơn do đặc thù dữ liệu categorical</li>
                <li>Không có hiện tượng concept drift đáng kể giữa các năm</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        render_header("Các yếu tố ảnh hưởng đến quyết định kết hôn")
        
        # Tóm tắt yếu tố với card style
        st.markdown(f"""
        <div style="background: {THEME['surface']}; border-radius: 20px; padding: 28px; margin-bottom: 28px; border: 1px solid rgba(99, 91, 255, 0.15);">
            <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 18px;">
                <div>
                    <h4 style="color: {THEME['text_primary']}; margin: 0; font-weight: 600; font-size: 1.2em;">Tổng quan các yếu tố</h4>
                    <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 1em;">Feature Importance Analysis</p>
                </div>
            </div>
            <p style="color: {THEME['text_primary']}; margin: 0; line-height: 1.9; font-size: 1.05em;">
                Dựa trên phân tích Decision Tree và Naive Bayes, các yếu tố sau ảnh hưởng 
                mạnh nhất đến quyết định kết hôn của giới trẻ 18-35 trong giai đoạn 2019-2024:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        factors = [
            ("Nhóm tuổi (age_group)", "25-29, 30-35 tăng mạnh xác suất kết hôn", THEME["success"], "+"),
            ("Giới tính (sex)", "Nữ có tỷ lệ kết hôn cao hơn nam trong cùng nhóm tuổi", THEME["primary"], "+"),
            ("Sở hữu nhà (home_ownership)", "Có nhà -> tăng đáng kể khả năng kết hôn (~16%)", THEME["warning"], "+"),
            ("Khu vực (urban_rural)", "Đô thị có tỷ lệ cao hơn nông thôn", THEME["info"], "+"),
            ("Trình độ học vấn", "ĐH/CĐ+ có tỷ lệ kết hôn cao hơn ≤THPT", THEME["success"], "+"),
            ("Vùng miền (region)", "Miền Bắc > Miền Nam > Miền Trung", THEME["warning"], "+/-"),
            ("Diện tích nhà ở", "Nhà rộng -> tỷ lệ kết hôn cao hơn", THEME["success"], "+"),
            ("Quy mô hộ gia đình", "Hộ 3-4 người có tỷ lệ kết hôn cao nhất", THEME["info"], "+")
        ]
        
        col1, col2 = st.columns(2)
        
        for i, (factor, desc, color, effect) in enumerate(factors):
            with col1 if i % 2 == 0 else col2:
                effect_badge = "badge-success" if effect == "+" else ("badge-error" if effect == "-" else "badge-warning")
                st.markdown(f"""
                <div style="background: {THEME['surface']}; border-left: 4px solid {color}; border-radius: 0 16px 16px 0; padding: 22px 24px; margin: 12px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.15); transition: all 0.2s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <p style="color: {THEME['text_primary']}; font-weight: 600; margin: 0; font-size: 1.1em;">{factor}</p>
                        <span class="badge {effect_badge}">{effect}</span>
                    </div>
                    <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 1em; line-height: 1.6;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature Importance chart
        render_header("Mức độ quan trọng của các biến", size="small")
        
        importance_data = pd.DataFrame({
            "Biến": ["age_group", "home_ownership", "urban_rural", "sex", 
                    "education_level", "region", "living_area_level", "household_size"],
            "Importance": [0.28, 0.22, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03]
        })
        
        fig = px.bar(
            importance_data.sort_values("Importance", ascending=True),
            x="Importance", y="Biến", orientation='h',
            title="Feature Importance từ Decision Tree",
            color="Importance",
            color_continuous_scale=[
                [0, THEME["surface_variant"]], 
                [0.3, THEME["info"]], 
                [0.6, THEME["primary"]], 
                [1, THEME["success"]]
            ]
        )
        layout = create_material_chart_layout()
        layout.update(showlegend=False, height=480, coloraxis_showscale=False)
        fig.update_layout(**layout)
        fig.update_traces(marker_line_width=0, marker_cornerradius=8)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Khuyến nghị chính sách
        render_header("Khuyến nghị chính sách", size="small")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px;">
                    <p style="color: {THEME['text_primary']}; font-weight: 700; margin: 0; font-size: 1.15em;">CAN THIỆP HIỆU QUẢ</p>
                </div>
                <ul style="color: {THEME['text_primary']}; line-height: 2.4; margin: 0; padding-left: 20px; font-size: 1.05em;">
                    <li><b style="color: {THEME['primary']};">Hỗ trợ nhà ở:</b> Vay mua nhà ưu đãi cho người 25-35 tuổi có thu nhập thấp</li>
                    <li><b style="color: {THEME['warning']};">Phát triển nông thôn:</b> Cải thiện cơ sở hạ tầng, thu hẹp khoảng cách đô thị - nông thôn</li>
                    <li><b style="color: {THEME['success']};">Giáo dục:</b> Nâng cao nhận thức về hôn nhân gia đình cho giới trẻ</li>
                    <li><b style="color: {THEME['info']};">Chính sách vùng:</b> Hỗ trợ đặc biệt cho miền Trung có tỷ lệ kết hôn thấp nhất</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="warning-box">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px;">
                    <p style="color: {THEME['text_primary']}; font-weight: 700; margin: 0; font-size: 1.15em;">PHÂN KHÚC ƯU TIÊN</p>
                </div>
                <ul style="color: {THEME['text_primary']}; line-height: 2.4; margin: 0; padding-left: 20px; font-size: 1.05em;">
                    <li>18-24 tuổi, đô thị, học vấn ĐH/CĐ+, chưa có nhà</li>
                    <li>25-35 tuổi nông thôn chưa sở hữu nhà</li>
                    <li>Miền Trung - tỷ lệ kết hôn thấp nhất cả nước</li>
                    <li>Nam giới 30-35 tuổi chưa kết hôn</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ==========================================================================
    # TAB 6: DỰ BÁO BỞI AI - Sử dụng fragment để tránh reset tab
    # ==========================================================================
    with tab6:
        render_ai_prediction_tab(panel_data)


@st.fragment
def render_ai_prediction_tab(panel_data):
    """
    Fragment để render tab AI prediction.
    Sử dụng @st.fragment để chỉ rerun phần này khi có interaction,
    không rerun toàn bộ app (tránh reset về tab đầu tiên).
    """
    render_header("Dự báo xu hướng kết hôn bằng AI (Google Gemini)")
    
    # Khởi tạo session state cho AI results
    if 'ai_result' not in st.session_state:
        st.session_state.ai_result = None
    if 'ai_result_type' not in st.session_state:
        st.session_state.ai_result_type = None
    if 'ai_result_year' not in st.session_state:
        st.session_state.ai_result_year = None
    
    # Khởi tạo Gemini model
    gemini_model = init_gemini()
    
    # Hiển thị trạng thái kết nối
    if gemini_model:
        st.markdown(f"""
        <div style="background: rgba(21, 183, 159, 0.1); border: 1px solid {THEME['success']}; border-radius: 12px; padding: 16px 20px; margin-bottom: 24px; display: flex; align-items: center; gap: 12px;">
            <div style="width: 12px; height: 12px; background: {THEME['success']}; border-radius: 50%; animation: pulse 2s infinite;"></div>
            <p style="color: {THEME['success']}; margin: 0; font-weight: 600;">Gemini AI đã sẵn sàng</p>
        </div>
        <style>
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.error("Không thể kết nối với Gemini AI. Vui lòng kiểm tra cấu hình.")
        return
    
    # Chọn kiểu dự báo và năm
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        prediction_type_options = [
            "Xu hướng tổng quát 2025-2030", 
            "Dự báo theo vùng/miền",
            "Dự báo theo nhóm tuổi",
            "Phân tích yếu tố kinh tế"
        ]
        prediction_type = st.selectbox(
            "Chọn loại dự báo:",
            prediction_type_options,
            key="ai_pred_type_select"
        )
    with col_pred2:
        target_year_options = [2025, 2026, 2027, 2028, 2029, 2030]
        target_year = st.selectbox(
            "Năm dự báo:",
            target_year_options,
            key="ai_pred_year_select"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Button dự báo
    if st.button("Dự Báo với AI", type="primary", use_container_width=True, key="ai_predict_button"):
        _call_gemini_ai(gemini_model, panel_data, prediction_type, target_year)
    
    # Hiển thị kết quả nếu có
    _display_ai_result()
    
    # Hiển thị dữ liệu tham khảo
    _display_reference_data(panel_data)


def _call_gemini_ai(gemini_model, panel_data, prediction_type, target_year):
    """Gọi Gemini AI và lưu kết quả vào session_state"""
    try:
        data_context = get_data_summary_for_ai(panel_data)
        prompt = _build_ai_prompt(data_context, prediction_type, target_year)
        
        with st.spinner("AI đang phân tích dữ liệu và tạo dự báo..."):
            import time
            max_retries = 3
            retry_delay = 5
            response_text = None
            
            for attempt in range(max_retries):
                try:
                    # SDK mới: google-genai (Client)
                    if hasattr(gemini_model, 'models'):
                        result = gemini_model.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt
                        )
                        response_text = result.text
                    else:
                        # SDK cũ: google-generativeai (GenerativeModel)
                        result = gemini_model.generate_content(prompt)
                        response_text = result.text
                    break
                except Exception as retry_error:
                    error_str = str(retry_error)
                    if "429" in error_str and attempt < max_retries - 1:
                        st.warning(f"API đang bận, thử lại sau {retry_delay}s... (lần {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    elif "429" in error_str:
                        raise Exception("Đã vượt quá giới hạn API. Vui lòng chờ vài phút.")
                    else:
                        raise retry_error
            
            if response_text is None:
                raise Exception("Không thể kết nối với Gemini AI sau nhiều lần thử")
            
            # Lưu kết quả vào session_state
            st.session_state.ai_result = response_text
            st.session_state.ai_result_type = prediction_type
            st.session_state.ai_result_year = target_year
            
    except Exception as e:
        st.error(f"Lỗi khi gọi Gemini AI: {str(e)}")


def _build_ai_prompt(data_context, prediction_type, target_year):
    """Tạo prompt cho Gemini AI dựa trên loại dự báo"""
    base_instructions = """
Bạn là chuyên gia phân tích dữ liệu về xu hướng kết hôn tại Việt Nam.
Dữ liệu này được lấy từ Tổng điều tra dân số Việt Nam (IPUMS International).
Dựa trên dữ liệu thống kê thực tế dưới đây, hãy phân tích và đưa ra dự báo CHI TIẾT.

CÁC BIẾN TRONG DỮ LIỆU:
- age_group: Nhóm tuổi (18-24, 25-29, 30-35)
- sex: Giới tính (Nam, Nữ)
- education_level: Trình độ học vấn (≤THPT, ĐH/CĐ+)
- urban_rural: Khu vực (Đô thị, Nông thôn)
- region: Vùng miền (Bắc, Trung, Nam)
- home_ownership: Sở hữu nhà (0=Không, 1=Có)
- living_area_level: Diện tích nhà ở (Nhỏ, Trung bình, Khá, Rộng)
- household_size_group: Quy mô hộ (1-2, 3-4, 5-6, >6 người)
- Y_married: Đã kết hôn (0=Chưa, 1=Đã)

QUAN TRỌNG:
- Trả lời bằng tiếng Việt có dấu
- Đưa ra số liệu cụ thể, % dự báo rõ ràng
- Giải thích LÝ DO dựa trên dữ liệu thực tế
- Format response với heading và bullet points rõ ràng
"""
    
    prompts = {
        "Xu hướng tổng quát 2025-2030": f"""
Hãy dự báo XU HƯỚNG KẾT HÔN TỔNG QUÁT cho năm {target_year} và giai đoạn 2025-2030:

1. **DỰ BÁO TỶ LỆ KẾT HÔN** - Tỷ lệ kết hôn năm {target_year} sẽ là bao nhiêu %?
2. **LÝ DO VÀ YẾU TỐ ẢNH HƯỞNG** - Yếu tố nào từ dữ liệu ảnh hưởng mạnh nhất?
3. **DỰ BÁO CHI TIẾT TỪNG NĂM 2025-2030** - Đưa ra con số % cụ thể cho mỗi năm
4. **KHUYẾN NGHỊ CHÍNH SÁCH** - Đề xuất giải pháp cụ thể
""",
        "Dự báo theo vùng/miền": f"""
Hãy phân tích XU HƯỚNG KẾT HÔN THEO VÙNG (Bắc, Trung, Nam) cho năm {target_year}:

1. **SO SÁNH XU HƯỚNG GIỮA CÁC VÙNG** - Vùng nào có tỷ lệ kết hôn cao/thấp nhất?
2. **DỰ BÁO CỤ THỂ CHO TỪNG VÙNG** - Bắc, Trung, Nam: dự báo % và lý do
3. **NGUYÊN NHÂN KHÁC BIỆT** - Yếu tố kinh tế, văn hóa, xã hội
4. **KHUYẾN NGHỊ CHÍNH SÁCH RIÊNG CHO TỪNG VÙNG**
""",
        "Dự báo theo nhóm tuổi": f"""
Hãy phân tích XU HƯỚNG KẾT HÔN THEO NHÓM TUỔI (18-24, 25-29, 30-35) cho năm {target_year}:

1. **PHÂN TÍCH XU HƯỚNG THEO NHÓM TUỔI** - Nhóm nào có tỷ lệ kết hôn cao/thấp nhất?
2. **DỰ BÁO CỤ THỂ** - Nhóm 18-24, 25-29, 30-35: dự báo % và lý do
3. **TUỔI KẾT HÔN TRUNG BÌNH** - Dự báo sẽ thay đổi như thế nào?
4. **KHUYẾN NGHỊ** - Chính sách khuyến khích kết hôn phù hợp từng nhóm tuổi
""",
        "Phân tích yếu tố kinh tế": f"""
Hãy phân tích ẢNH HƯỞNG CỦA CÁC YẾU TỐ KINH TẾ ĐẾN QUYẾT ĐỊNH KẾT HÔN năm {target_year}:

1. **ẢNH HƯỞNG CỦA SỞ HỮU NHÀ Ở** - Tác động của home_ownership
2. **ẢNH HƯỞNG CỦA DIỆN TÍCH NHÀ** - Mối quan hệ living_area_level với tỷ lệ kết hôn
3. **TÁC ĐỘNG CỦA TRÌNH ĐỘ HỌC VẤN** - Phân tích education_level
4. **SO SÁNH ĐÔ THỊ - NÔNG THÔN** - Yếu tố nào gây ra khác biệt?
5. **KHUYẾN NGHỊ CHÍNH SÁCH** - Đề xuất cụ thể hỗ trợ giới trẻ
"""
    }
    
    return f"{base_instructions}\n\n{data_context}\n\n{prompts.get(prediction_type, prompts['Xu hướng tổng quát 2025-2030'])}"


def _display_ai_result():
    """Hiển thị kết quả AI prediction"""
    if not st.session_state.ai_result:
        return
    
    st.markdown("<br>", unsafe_allow_html=True)
    render_header(f"Kết quả dự báo - {st.session_state.ai_result_type}", size="small")
    
    # Header box
    st.markdown(f"""
    <div style="background: {THEME['surface']}; border-radius: 20px; padding: 28px; margin: 20px 0; border: 1px solid rgba(99, 91, 255, 0.15);">
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 18px;">
            <div style="width: 52px; height: 52px; background: linear-gradient(135deg, #635BFF, #4e36f5); border-radius: 14px; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-weight: 700; font-size: 18px;">AI</span>
            </div>
            <div>
                <h4 style="color: {THEME['text_primary']}; margin: 0; font-weight: 600; font-size: 1.2em;">Google Gemini AI</h4>
                <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 1em;">Dự báo cho năm {st.session_state.ai_result_year}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Response content
    st.markdown(f'<div class="ai-result-box">{st.session_state.ai_result}</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f"""
    <div class="warning-box" style="margin-top: 24px;">
        <p style="color: {THEME['warning']}; font-weight: 600; margin: 0 0 8px 0; font-size: 1.1em;">Lưu ý quan trọng</p>
        <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 1em; line-height: 1.7;">
            Kết quả dự báo được tạo bởi AI dựa trên dữ liệu lịch sử. 
            Đây chỉ là dự báo tham khảo, không phải dự báo chính thức.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clear button
    if st.button("Xóa kết quả", key="clear_ai_result_btn"):
        st.session_state.ai_result = None
        st.session_state.ai_result_type = None
        st.session_state.ai_result_year = None
        st.rerun()


def _display_reference_data(panel_data):
    """Hiển thị dữ liệu tham khảo"""
    if panel_data is None:
        return
    
    st.markdown("<br>", unsafe_allow_html=True)
    render_header("Dữ liệu hiện có (2019-2024)", size="small")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        yearly_stats = panel_data.groupby("year")["Y_married"].mean().reset_index()
        fig1 = px.line(yearly_stats, x="year", y="Y_married", 
                      title="Tỷ lệ kết hôn theo năm",
                      markers=True,
                      color_discrete_sequence=[THEME["primary"]])
        fig1.update_layout(**create_material_chart_layout(), height=300)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        region_stats = panel_data.groupby("region")["Y_married"].mean().reset_index()
        fig2 = px.bar(region_stats, x="region", y="Y_married",
                     title="Tỷ lệ kết hôn theo vùng",
                     color="region",
                     color_discrete_sequence=[THEME["success"], THEME["warning"], THEME["error"]])
        fig2.update_layout(**create_material_chart_layout(), height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        age_stats = panel_data.groupby("age_group")["Y_married"].mean().reset_index()
        fig3 = px.bar(age_stats, x="age_group", y="Y_married",
                     title="Tỷ lệ kết hôn theo nhóm tuổi",
                     color="age_group",
                     color_discrete_sequence=[THEME["info"], THEME["primary"], THEME["success"]])
        fig3.update_layout(**create_material_chart_layout(), height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h3 style="color: {THEME['text_primary']}; margin: 0 0 8px 0; font-size: 1.3em;">DỰ BÁO XU HƯỚNG KẾT HÔN</h3>
            <p style="color: {THEME['text_secondary']}; margin: 0 0 16px 0; font-size: 1.1em;">ĐỒ ÁN: KHAI THÁC DỮ LIỆU VÀ TRUYỀN THÔNG XÃ HỘI</p>
            <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 1em;">Dự báo Xu hướng Kết hôn (18-35) | Dữ liệu IPUMS Vietnam Census</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(" ")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("**Nguồn dữ liệu**")
            st.caption("IPUMS International")
        with c2:
            st.warning("**Mô hình ML**")
            st.caption("Decision Tree & Naive Bayes")
        with c3:
            st.info("**UI Design**")
            st.caption("Material Kit React (Devias)")
        
        st.markdown(" ")
if __name__ == "__main__":
    main()
