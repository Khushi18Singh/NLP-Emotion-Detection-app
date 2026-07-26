import streamlit as st
import pickle
import json
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import seaborn as sns
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ----------------------------------------------------------------------------
# Setup & NLTK
# ----------------------------------------------------------------------------
try:
    stopwords.words('english')
except:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

st.set_page_config(
    page_title="Khushi's Emotion Detection Platform",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

emotion_mapping = {
    0: 'Sadness 😔',
    1: 'Anger 😡',
    2: 'Love ❤️',
    3: 'Surprise 😲',
    4: 'Fear 😨',
    5: 'Joy 😊',
}

# Dictionary for Hinglish -> English translation
HINGLISH_MAP = {
    "khush": "happy joy", "khushi": "happiness joy", "bohot khush": "very happy",
    "mast": "awesome happy", "badiya": "great good", "shaandar": "fantastic",
    "gussa": "angry anger", "gusse": "angry anger", "naarat": "angry",
    "udaas": "sad sadness", "udaasi": "sadness", "dukhi": "sad unhappy", "dukh": "sorrow sad",
    "pyaar": "love", "pyar": "love", "ishq": "love", "mohabbat": "love",
    "darr": "scared fear", "dar": "scared fear", "khauf": "fear horror", "bhaya": "scared",
    "hairan": "surprised amazed", "aashcharya": "surprised amazed", "chawk": "surprised",
    "bura": "bad sad", "ganda": "terrible bad", "bekar": "useless bad", "durdasha": "sad",
    "aaj": "today", "kal": "tomorrow", "mai": "i", "main": "i", "mujhe": "me", "mera": "my", "meri": "my"
}

# Dynamic Reactive UI CSS Themes per Emotion (Predict Page)
EMOTION_THEMES = {
    'Sadness 😔': """<style>
        body, .stApp { background: radial-gradient(circle at top left, #172554 0%, #0f172a 60%, #07090e 100%) !important; }
        h1 { background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    </style>""",
    'Anger 😡': """<style>
        body, .stApp { background: radial-gradient(circle at top left, #450a0a 0%, #180505 60%, #07090e 100%) !important; }
        h1 { background: linear-gradient(135deg, #f87171 0%, #ef4444 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    </style>""",
    'Love ❤️': """<style>
        body, .stApp { background: radial-gradient(circle at top left, #500724 0%, #1f0410 60%, #07090e 100%) !important; }
        h1 { background: linear-gradient(135deg, #f472b6 0%, #ec4899 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    </style>""",
    'Surprise 😲': """<style>
        body, .stApp { background: radial-gradient(circle at top left, #3b0764 0%, #17032a 60%, #07090e 100%) !important; }
        h1 { background: linear-gradient(135deg, #c084fc 0%, #8b5cf6 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    </style>""",
    'Fear 😨': """<style>
        body, .stApp { background: radial-gradient(circle at top left, #064e3b 0%, #022c22 60%, #07090e 100%) !important; }
        h1 { background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    </style>""",
    'Joy 😊': """<style>
        body, .stApp { background: radial-gradient(circle at top left, #451a03 0%, #1a0b02 60%, #07090e 100%) !important; }
        h1 { background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    </style>"""
}

# Base Aesthetic Styling
st.markdown("""
<style>
    body, .stApp {
        background: radial-gradient(circle at top left, #1a1e2c 0%, #0d1017 60%, #07090e 100%) !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    .stApp p, .stApp span, .stApp li, .stApp label { color: #e2e8f0 !important; }
    .stMarkdownContainer p, .stMarkdown p { color: #e2e8f0 !important; font-size: 1.05rem !important; line-height: 1.6 !important; }
    strong, b { color: #ffffff !important; font-weight: 700 !important; }
    .stCaption, [data-testid="stCaptionContainer"] p { color: #cbd5e1 !important; font-size: 0.95rem !important; font-weight: 500 !important; }

    h1 {
        background: linear-gradient(135deg, #c084fc 0%, #60a5fa 50%, #38bdf8 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-size: 2.5rem !important; font-weight: 800 !important; padding-bottom: 0.3rem !important;
    }
    h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 700 !important; letter-spacing: -0.01em !important; }

    /* Highlighted Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121624 0%, #0d101a 100%) !important;
        border-right: 1px solid #3b82f644 !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.4) !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1 0%, #3b82f6 100%) !important;
        color: #ffffff !important; border: 1px solid #4f46e5 !important;
        border-radius: 8px !important; padding: 0.6rem 1.4rem !important; font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35) !important; transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:first-child:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 18px rgba(99, 102, 241, 0.5) !important; border-color: #818cf8 !important; }

    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important; border: none !important; border-radius: 8px !important;
        font-weight: 600 !important; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px !important; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important; padding: 10px 22px !important; font-weight: 600 !important;
        background-color: #161a26 !important; color: #94a3b8 !important; border: 1px solid #282f42 !important; border-bottom: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6366f1 0%, #3b82f6 100%) !important;
        color: #ffffff !important; border-color: #6366f1 !important;
    }

    /* Highlighted Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 40, 0.95) !important;
        border: 1px solid #3b82f644 !important;
        border-radius: 12px !important; padding: 14px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3) !important;
    }
    div[data-testid="stMetric"] label { color: #cbd5e1 !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; }

    div[data-testid="stExpander"] { background-color: #141824 !important; border: 1px solid #282f42 !important; border-radius: 10px !important; }
    div[data-testid="stExpander"] summary { color: #f1f5f9 !important; font-weight: 600 !important; }
    
    /* High-Visibility Input Text & Placeholder Text for Search Bars & Text Areas */
    textarea, input[type="text"], [data-baseweb="input"] input {
        background-color: #141824 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1.5px solid #38bdf855 !important;
        border-radius: 10px !important;
    }
    ::placeholder,
    input::placeholder,
    textarea::placeholder,
    [data-baseweb="input"] input::placeholder,
    [data-baseweb="textarea"] textarea::placeholder {
        color: #cbd5e1 !important;
        opacity: 0.95 !important;
        font-weight: 500 !important;
    }

    /* High-Visibility Button Labels */
    .stButton > button, button[kind="primary"], button[kind="secondary"] {
        font-weight: 800 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
    }
    /* Super Visible & Glowing Floating Sidebar Toggle Button (>>, >, <) at Left Corner */
    button[data-testid="stSidebarCollapseButton"], 
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button, 
    button[kind="header"],
    section[data-testid="stSidebar"] button[kind="header"] {
        background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%) !important;
        color: #ffffff !important;
        border: 2px solid #ffffff !important;
        border-radius: 12px !important;
        box-shadow: 0 0 24px rgba(56, 189, 248, 1), 0 0 10px rgba(99, 102, 241, 0.8) !important;
        padding: 6px 12px !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 99999999 !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover, 
    [data-testid="collapsedControl"] button:hover {
        transform: scale(1.15) !important;
        box-shadow: 0 0 32px rgba(56, 189, 248, 1), 0 0 16px rgba(255, 255, 255, 0.9) !important;
    }
    button[data-testid="stSidebarCollapseButton"] svg, 
    [data-testid="collapsedControl"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        stroke: #ffffff !important;
        width: 26px !important;
        height: 26px !important;
        filter: drop-shadow(0 0 6px rgba(255,255,255,0.95)) !important;
    }

    /* Theme-Matching File Uploader & Choose File Button */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #161a29 0%, #0f131f 100%) !important;
        border: 2px dashed #38bdf888 !important;
        border-radius: 14px !important;
        color: #f1f5f9 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 22px rgba(56, 189, 248, 0.6) !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] button,
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1.5px solid #ffffff !important;
        border-radius: 10px !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.8) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploader"] button:hover,
    button[data-testid="baseButton-secondary"]:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 26px rgba(56, 189, 248, 1) !important;
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    /* Theme-Matching Selectbox, Text Area & Buttons */
    div[data-baseweb="select"] > div {
        background-color: #161a29 !important;
        border: 1.5px solid #38bdf844 !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.4) !important;
    }
    button[kind="secondary"] {
        background: #161a29 !important;
        color: #38bdf8 !important;
        border: 1.5px solid #38bdf866 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover {
        background: #38bdf8 !important;
        color: #0f172a !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.7) !important;
    }

    /* Theme-Matching Dataframe & Table Containers */
    div[data-testid="stDataFrame"],
    [data-testid="stTable"],
    .stDataFrame {
        background: linear-gradient(135deg, #161a29 0%, #0f131f 100%) !important;
        border: 1.5px solid #38bdf855 !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.45) !important;
        padding: 4px !important;
    }
    div[data-testid="stDataFrame"] [data-testid="stTable"] {
        background-color: transparent !important;
    }
    .stDataFrame th, [data-testid="stTable"] th, th {
        background: #1e2538 !important;
        color: #38bdf8 !important;
        font-weight: 800 !important;
        font-size: 0.92rem !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid #38bdf866 !important;
    }
    .stDataFrame td, [data-testid="stTable"] td, td {
        background-color: #141824 !important;
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
        border-bottom: 1px solid #282f42 !important;
    }
    /* Streamlit Top Header Bar — Glowing Brand Badge & Deploy Button Styling */
    header[data-testid="stHeader"],
    .stAppHeader {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        background: linear-gradient(90deg, rgba(13, 16, 26, 0.95) 0%, rgba(22, 27, 40, 0.95) 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1.5px solid #38bdf866 !important;
        padding: 0 18px !important;
    }
    header[data-testid="stHeader"]::before {
        content: "Khushi's 🎭 Emotion Detection Platform";
        background: linear-gradient(135deg, #c084fc 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.02em !important;
        white-space: nowrap !important;
        margin-left: 55px !important;
        text-shadow: 0 0 16px rgba(56, 189, 248, 0.4) !important;
    }
    header[data-testid="stHeader"] button,
    .stAppHeader button {
        color: #ffffff !important;
        fill: #ffffff !important;
        background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%) !important;
        border: 1.5px solid #ffffff !important;
        border-radius: 10px !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.8) !important;
        font-weight: 700 !important;
    }
    header[data-testid="stHeader"] button:hover,
    .stAppHeader button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 24px rgba(56, 189, 248, 1) !important;
    }

    /* Bright Glowing Streamlit Running Status Widget & Top Header Labels */
    [data-testid="stStatusWidget"],
    .stStatusWidget,
    [data-testid="stStatusWidget"] *,
    header[data-testid="stHeader"] p,
    header[data-testid="stHeader"] span,
    header[data-testid="stHeader"] label,
    .stAppHeader p,
    .stAppHeader span,
    .stAppHeader label {
        color: #38bdf8 !important;
        fill: #38bdf8 !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.9) !important;
    }
    [data-testid="stStatusWidget"] svg,
    .stStatusWidget svg {
        fill: #38bdf8 !important;
        stroke: #38bdf8 !important;
        filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.9)) !important;
    }

    /* Plotly Modebar & Fullscreen/Zoom Buttons — Bright Light Accent Button with Dark Contrast Icon */
    .modebar {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 10px !important;
        padding: 4px 6px !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.7) !important;
    }
    .modebar-btn {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
        border: 1.5px solid #ffffff !important;
        border-radius: 8px !important;
        margin: 0 2px !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.9) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .modebar-btn svg {
        fill: #0f172a !important;
        color: #0f172a !important;
        stroke: #0f172a !important;
        width: 18px !important;
        height: 18px !important;
        filter: drop-shadow(0 0 2px rgba(0,0,0,0.5)) !important;
    }
    .modebar-btn:hover {
        background: #ffffff !important;
        transform: scale(1.12) !important;
        box-shadow: 0 0 22px rgba(255, 255, 255, 1) !important;
    }
    .modebar-btn:hover svg {
        fill: #0284c7 !important;
        color: #0284c7 !important;
        stroke: #0284c7 !important;
    }

    .theory-card {
        background: linear-gradient(135deg, #181d2c 0%, #111522 100%) !important;
        border: 1.5px solid #3b82f644 !important;
        border-left: 5px solid #38bdf8 !important;
        border-radius: 14px !important;
        padding: 24px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
    }
    .theory-card h3, .theory-card h4 {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        margin-top: 0 !important;
        margin-bottom: 14px !important;
    }
    .theory-card p, .theory-card li {
        color: #f1f5f9 !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }

    .flowchart-frame {
        background: linear-gradient(135deg, rgba(22, 27, 39, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1.5px solid #38bdf866;
        border-radius: 16px;
        padding: 16px;
        margin-top: 14px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }

    .rephrase-box {
        background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 10px;
        padding: 16px; margin-top: 12px; margin-bottom: 12px;
    }

    .developer-badge {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(56, 189, 248, 0.15) 100%);
        border: 1px solid #38bdf888;
        border-radius: 12px;
        padding: 14px;
        margin-top: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Text Cleaning & Translation Functions
# ----------------------------------------------------------------------------
def translate_hinglish(text):
    words = text.lower().split()
    translated = [HINGLISH_MAP.get(w, w) for w in words]
    return ' '.join(translated)

def remove_pucn(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(txt):
    return ''.join(i for i in txt if not i.isdigit())

def remove_emojis(txt):
    return ''.join(i for i in txt if i.isascii())

def remove_stopwords(txt):
    words = txt.split()
    cleaned = [i for i in words if i not in stop_words]
    return ' '.join(cleaned)

def clean_text(text, input_lang="🌐 Auto-Detect / English"):
    if "Hinglish" in input_lang or any(w in text.lower().split() for w in HINGLISH_MAP.keys()):
        text = translate_hinglish(text)
    text = text.lower()
    text = remove_pucn(text)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text

def generate_rephrase(emotion, raw_text):
    if emotion == 'Anger 😡':
        return f"💡 **Calm Alternative:** \"I am feeling frustrated regarding: '{raw_text[:60]}...', but I would like to resolve this constructively.\""
    elif emotion == 'Sadness 😔':
        return f"💡 **Positive Alternative:** \"I am experiencing a low moment about: '{raw_text[:60]}...', but I am remaining hopeful and taking things one step at a time.\""
    elif emotion == 'Fear 😨':
        return f"💡 **Courageous Alternative:** \"I am feeling uncertain about: '{raw_text[:60]}...', but I am focusing on what I can control with confidence.\""
    return None

def generate_html_report(df_results):
    total = len(df_results)
    emo_counts = df_results["Predicted Emotion"].value_counts().to_dict()
    rows_html = ""
    for idx, row in df_results.head(50).iterrows():
        rows_html += f"<tr style='background:#141824; color:#f1f5f9;'><td style='padding:10px;border:1px solid #38bdf844;'>{row.get('Text', 'N/A')}</td><td style='padding:10px;border:1px solid #38bdf844;color:#38bdf8;font-weight:bold;'>{row.get('Predicted Emotion', 'N/A')}</td><td style='padding:10px;border:1px solid #38bdf844;color:#4ade80;font-weight:bold;'>{row.get('Confidence %', 'N/A')}%</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Khushi's Emotion Analytics Report</title></head>
    <body style="font-family: Arial, sans-serif; padding: 30px; background: #0b0d14; color: #f1f5f9;">
        <h1 style="color: #38bdf8; background: linear-gradient(135deg, #38bdf8 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Khushi's 🎭 Emotion Detection Executive Analytics Report</h1>
        <hr style="border-color: #38bdf844;"/>
        <h3 style="color: #c084fc;">📌 Executive Summary</h3>
        <p><strong>Lead Developer:</strong> Khushi Singh</p>
        <p><strong>Total Sample Records Processed:</strong> {total}</p>
        <h4 style="color: #38bdf8;">📊 Class Distribution Breakdown:</h4>
        <ul style="line-height: 1.8;">
            {"".join([f"<li><strong style='color:#38bdf8;'>{k}:</strong> {v} records ({v/total*100:.1f}%)</li>" for k, v in emo_counts.items()])}
        </ul>
        <hr style="border-color: #38bdf844;"/>
        <h3 style="color: #4ade80;">📋 Sample Predictions Table (Top 50)</h3>
        <table style="width:100%; border-collapse: collapse; background: #161a29; border: 1.5px solid #38bdf866; border-radius: 10px;">
            <thead>
                <tr style="background:#1e2538; color:#38bdf8;">
                    <th style="padding:12px; border:1px solid #38bdf844; text-align:left;">Text</th>
                    <th style="padding:12px; border:1px solid #38bdf844; text-align:left;">Predicted Emotion</th>
                    <th style="padding:12px; border:1px solid #38bdf844; text-align:left;">Confidence</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <footer style="margin-top:40px; color:#94a3b8; font-size:0.85rem;">Generated by Khushi's Emotion Detection Platform</footer>
    </body>
    </html>
    """
    return html_content

# ----------------------------------------------------------------------------
# Load model artifacts & scores
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

@st.cache_data
def load_scores():
    try:
        with open('model_scores.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

vectorizer, model = load_models()
scores = load_scores()

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------
# Helper: Interactive Sidebar Plotly Accuracy Chart
# ----------------------------------------------------------------------------
def create_sidebar_acc_chart(acc_dict):
    import plotly.express as px

    sorted_items = sorted(acc_dict.items(), key=lambda x: x[1])
    models = [item[0] for item in sorted_items]
    accuracies = [round(item[1] * 100, 2) for item in sorted_items]

    short_labels = []
    for m in models:
        if "Logistic" in m:
            short_labels.append("Logistic Reg 🏆")
        elif "Bag of Words" in m:
            short_labels.append("Naive Bayes (BoW)")
        else:
            short_labels.append("Naive Bayes (TF-IDF)")

    df_plot = pd.DataFrame({
        "Model": short_labels,
        "Accuracy %": accuracies,
        "Full Name": models
    })

    fig = px.bar(
        df_plot,
        x="Accuracy %",
        y="Model",
        orientation="h",
        text="Accuracy %",
        hover_data=["Full Name", "Accuracy %"],
        color="Accuracy %",
        color_continuous_scale=["#334155", "#6366f1", "#38bdf8"]
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        marker_line_color='#818cf8',
        marker_line_width=1.2,
        hovertemplate='<b>%{customdata[0]}</b><br>Accuracy: <b>%{x:.2f}%</b><extra></extra>'
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=40, t=10, b=10),
        height=210,
        showlegend=False,
        coloraxis_showscale=False,
        font=dict(color="#f1f5f9", size=11),
        xaxis=dict(showgrid=False, visible=False, range=[0, 115]),
        yaxis=dict(showgrid=False, tickfont=dict(color="#f1f5f9", size=10.5))
    )
    return fig

# Helper: Interactive Confidence Plotly Bar Chart
def create_confidence_plotly_chart(prob_df):
    import plotly.express as px
    df = prob_df.reset_index()
    df["Confidence %"] = (df["Confidence"] * 100).round(1)

    fig = px.bar(
        df,
        x="Emotion",
        y="Confidence %",
        text="Confidence %",
        color="Confidence %",
        color_continuous_scale=["#334155", "#6366f1", "#38bdf8"]
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        marker_line_color='#818cf8',
        marker_line_width=1.2,
        hovertemplate='<b>%{x}</b><br>Confidence: <b>%{y:.1f}%</b><extra></extra>'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
        showlegend=False,
        coloraxis_showscale=False,
        font=dict(color="#f1f5f9", size=12),
        xaxis=dict(showgrid=False, tickfont=dict(color="#f1f5f9", size=11)),
        yaxis=dict(showgrid=True, gridcolor='#282f42', range=[0, 115], tickfont=dict(color="#f1f5f9", size=10.5))
    )
    return fig

# Helper: Multi-Dimensional Plotly Emotion Radar / Spider Chart
def create_emotion_radar_chart(probabilities):
    import plotly.graph_objects as go
    categories = [emotion_mapping[i].split()[0] for i in range(len(probabilities))]
    values = [round(p * 100, 1) for p in probabilities]

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.25)',
        line=dict(color='#38bdf8', width=2.5),
        marker=dict(size=8, color='#c084fc'),
        hovertemplate='<b>%{theta}</b>: <b>%{r:.1f}%</b><extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(15, 23, 42, 0.85)',
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True, tickfont=dict(color='#cbd5e1', size=9), gridcolor='#282f42'),
            angularaxis=dict(tickfont=dict(color='#f1f5f9', size=11, weight='bold'), gridcolor='#282f42')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=25, b=25),
        height=280,
        showlegend=False
    )
    return fig

# Helper: SpeechRecognition Audio Transcription
def transcribe_audio_file(audio_file):
    import speech_recognition as sr
    import tempfile
    r = sr.Recognizer()

    try:
        ext = audio_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        with sr.AudioFile(tmp_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text, None
    except sr.UnknownValueError:
        return None, "Audio speech could not be recognized. Please ensure clear speech and minimal background noise."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        return None, f"Audio processing format notice: {e}. Please test with a clean .wav or .mp3 file."

# Helper: PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
        
        # Split into non-empty paragraphs
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 15]
        if not paragraphs:
            paragraphs = [p.strip() for p in full_text.split('\n') if len(p.strip()) > 15]
        return paragraphs, None
    except Exception as e:
        return None, f"PDF extraction error: {e}"

# Helper: AI Mood Copilot & Wellness Advice
def generate_mood_copilot_wellness(emotion):
    wellness_map = {
        'Sadness 😔': {
            'action': "🧘 **Mindful Self-Care:** Take 5 deep breaths, step outside for fresh air, and engage in gentle journaling.",
            'quote': "✨ \"Tough times never last, but tough people do.\" — Robert H. Schuller",
            'copilot': "💙 **Copilot Tip:** Acknowledge your feelings without judgment. Reach out to a trusted friend or mentor."
        },
        'Anger 😡': {
            'action': "🌊 **4-7-8 Breathing Technique:** Inhale for 4s, hold for 7s, exhale slowly for 8s to calm your nervous system.",
            'quote': "✨ \"For every minute you remain angry, you give up sixty seconds of peace of mind.\" — Ralph Waldo Emerson",
            'copilot': "🔥 **Copilot Tip:** Take a 10-minute walk before responding to challenging emails or messages."
        },
        'Fear 😨': {
            'action': "🛡️ **Grounding 5-4-3-2-1 Exercise:** Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
            'quote': "✨ \"You gain strength, courage, and confidence by every experience in which you stop to look fear in the face.\" — Eleanor Roosevelt",
            'copilot': "💪 **Copilot Tip:** Focus strictly on actionable steps within your direct control."
        },
        'Love ❤️': {
            'action': "💖 **Spread Connection:** Express gratitude to someone who made a positive impact on your life today.",
            'quote': "✨ \"Love and kindness are never wasted. They always make a difference.\" — Barbara De Angelis",
            'copilot': "🌟 **Copilot Tip:** Channel your warmth into creative projects or supportive teamwork."
        },
        'Joy 😊': {
            'action': "🎉 **Savor the Moment:** Capture this positive energy and share your accomplishments with loved ones!",
            'quote': "✨ \"Joy is not in things; it is in us.\" — Richard Wagner",
            'copilot': "🚀 **Copilot Tip:** Use this peak motivation state to tackle ambitious goals!"
        },
        'Surprise 😲': {
            'action': "⚡ **Pause & Process:** Take a moment to absorb new information and evaluate fresh opportunities.",
            'quote': "✨ \"Life is full of surprises and serendipitous discoveries.\" — Anonymous",
            'copilot': "💡 **Copilot Tip:** Turn unexpected news into valuable learning experiences!"
        }
    }
    return wellness_map.get(emotion, None)

# ----------------------------------------------------------------------------
# Sidebar — Highlighted & Always Visible
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Model Comparison")
    if scores and "accuracy" in scores:
        # Interactive Plotly horizontal bar chart for sidebar with Fullscreen button only
        fig_acc = create_sidebar_acc_chart(scores["accuracy"])
        st.plotly_chart(
            fig_acc,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': [
                    'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian'
                ]
            }
        )

        for model_name, acc in scores["accuracy"].items():
            st.metric(label=model_name, value=f"{acc*100:.1f}%")
        st.caption("✅ **Logistic Regression (TF-IDF)** chosen — highest accuracy (86.3%).")
    else:
        st.info("`model_scores.json` not found.")

    st.markdown("---")
    st.markdown("### 👩‍💻 Developer Info")
    st.markdown("""
    <div class="developer-badge">
        <div style="color: #38bdf8; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Lead Developer</div>
        <div style="background: linear-gradient(135deg, #38bdf8 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 1.4rem; margin-top: 2px;">Khushi Singh</div>
        <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px; font-weight: 600;">NLP & ML Engineering</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Main Title & Top Interactive Glowing Navbar Banner
# ----------------------------------------------------------------------------
# Top Executive Dark Glass Hero Navbar Banner
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(22, 27, 40, 0.95) 0%, rgba(13, 16, 26, 0.95) 100%); border: 1.5px solid #38bdf866; border-radius: 20px; padding: 22px 28px; margin-bottom: 24px; box-shadow: 0 10px 35px rgba(0, 0, 0, 0.6);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; margin-bottom: 16px;">
        <div>
            <h1 style="background: linear-gradient(135deg, #c084fc 0%, #60a5fa 50%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.1rem; font-weight: 900; margin: 0; padding: 0; white-space: nowrap;">Khushi's 🎭 Emotion Detection Platform</h1>
            <div style="color: #cbd5e1; margin-top: 4px; font-size: 1.02rem; font-weight: 500;">Multi-Class NLP Machine Learning Analytics Engine</div>
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; border-top: 1px solid #38bdf833; padding-top: 16px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: rgba(99, 102, 241, 0.25); border: 1.5px solid #6366f1; border-radius: 12px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 0 14px rgba(99, 102, 241, 0.5);">👩‍💻</div>
            <div>
                <div style="color: #cbd5e1; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Lead Developer</div>
                <div style="background: linear-gradient(135deg, #38bdf8 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 1.2rem;">Khushi Singh</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: rgba(56, 189, 248, 0.25); border: 1.5px solid #38bdf8; border-radius: 12px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 0 14px rgba(56, 189, 248, 0.5);">⚡</div>
            <div>
                <div style="color: #cbd5e1; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Winning Model</div>
                <div style="color: #38bdf8; font-weight: 800; font-size: 1.15rem;">Logistic Regression</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: rgba(74, 222, 128, 0.25); border: 1.5px solid #4ade80; border-radius: 12px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 0 14px rgba(74, 222, 128, 0.5);">🎯</div>
            <div>
                <div style="color: #cbd5e1; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Benchmark Accuracy</div>
                <div style="color: #4ade80; font-weight: 800; font-size: 1.15rem;">86.28%</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: rgba(192, 132, 252, 0.25); border: 1.5px solid #c084fc; border-radius: 12px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 0 14px rgba(192, 132, 252, 0.5);">🎭</div>
            <div>
                <div style="color: #cbd5e1; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Emotion Classes</div>
                <div style="color: #c084fc; font-weight: 800; font-size: 1.15rem;">6 Categories</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["🔮 Predict", "📁 Batch Predict", "📊 Insights", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# Tab 1: Predict
# ----------------------------------------------------------------------------
with tab_predict:
    input_mode = st.radio("Choose Input Mode:", options=["📝 Text Input", "🎙️ Voice Audio Input"], horizontal=True, key="predict_input_mode")

    col_in1, col_in2 = st.columns([3, 1])

    with col_in2:
        input_lang = st.selectbox("🌐 Input Language:", options=["🌐 Auto-Detect / English", "🇮🇳 Hinglish / Hindi"], key="lang_select")

    user_input = ""

    with col_in1:
        if input_mode == "📝 Text Input":
            user_input = st.text_area(
                "Enter your text below:",
                height=130,
                placeholder="E.g., I feel so happy and excited! OR 'main aaj bohot khush hu'",
            )
        else:
            audio_file = st.file_uploader("🎙️ Upload Voice Audio File (.wav, .mp3):", type=["wav", "mp3"], key="audio_uploader")
            if audio_file is not None:
                with st.spinner("🎙️ Transcribing audio speech to text..."):
                    transcribed_text, audio_err = transcribe_audio_file(audio_file)
                    if transcribed_text:
                        user_input = transcribed_text
                        st.success(f"🎙️ **Transcribed Audio Speech:** \"{user_input}\"")
                    else:
                        st.warning(audio_err)

    col1, col2 = st.columns([1, 1])
    predict_clicked = col1.button("Predict Emotion", type="primary")
    clear_clicked = col2.button("Clear History")

    if clear_clicked:
        st.session_state.history = []
        st.success("History cleared!")
        st.rerun()

    if predict_clicked:
        if user_input.strip() == "":
            st.warning("Please enter text or upload audio speech to predict.")
        else:
            with st.spinner("Analyzing emotion..."):
                translated_text = translate_hinglish(user_input) if ("Hinglish" in input_lang or any(w in user_input.lower().split() for w in HINGLISH_MAP.keys())) else user_input
                cleaned_input = clean_text(user_input, input_lang)
                vectorized_input = vectorizer.transform([cleaned_input])

                prediction = model.predict(vectorized_input)[0]
                probabilities = model.predict_proba(vectorized_input)[0]
                conf_val = probabilities[prediction] * 100
                predicted_emo_str = emotion_mapping[prediction]

                st.markdown(EMOTION_THEMES.get(predicted_emo_str, ""), unsafe_allow_html=True)
                st.success(f"Predicted Emotion: **{predicted_emo_str}** (Confidence: **{conf_val:.1f}%**)")

                rephrase_msg = generate_rephrase(predicted_emo_str, user_input)
                if rephrase_msg:
                    st.markdown(f'<div class="rephrase-box">{rephrase_msg}</div>', unsafe_allow_html=True)

                # Charts Section: Bar Chart + Multi-Dimensional Radar Chart side-by-side
                c_chart1, c_chart2 = st.columns([1, 1])

                with c_chart1:
                    st.markdown("**📊 Confidence Distribution Bar Chart**")
                    prob_df = pd.DataFrame({
                        "Emotion": [emotion_mapping[i] for i in range(len(probabilities))],
                        "Confidence": probabilities,
                    }).set_index("Emotion")
                    fig_prob = create_confidence_plotly_chart(prob_df)
                    st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

                with c_chart2:
                    st.markdown("**🕸️ Multi-Dimensional Emotion Radar Chart**")
                    fig_radar = create_emotion_radar_chart(probabilities)
                    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

                # AI Mood Copilot & Mental Wellness Advice
                copilot_info = generate_mood_copilot_wellness(predicted_emo_str)
                if copilot_info:
                    st.markdown(f"""
                    <div class="theory-card" style="border-left: 6px solid #a855f7; margin-top: 14px;">
                        <h4 style="color: #c084fc !important; margin-bottom: 8px;">🤖 AI Mood Copilot & Wellness Advice</h4>
                        <p>{copilot_info['action']}</p>
                        <p>{copilot_info['quote']}</p>
                        <p>{copilot_info['copilot']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("🔍 Show Preprocessing & Translation Details"):
                    st.write("**Original Text:**", user_input)
                    if translated_text != user_input:
                        st.write("**Translated / Standardized Text:**", translated_text)
                    st.write("**Cleaned Feature Tokens:**", cleaned_input)

                st.session_state.history.insert(0, {
                    "Text": user_input,
                    "Predicted Emotion": predicted_emo_str,
                    "Confidence %": round(conf_val, 1),
                })

    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 🕒 Prediction History & Interactive Filters")
        hist_df = pd.DataFrame(st.session_state.history)

        f_col1, f_col2, f_col3 = st.columns([1.2, 1.5, 1.5])
        with f_col1:
            filter_options = ["All Emotions"] + sorted(hist_df["Predicted Emotion"].unique().tolist())
            history_emotion_filter = st.selectbox("Filter by Emotion:", options=filter_options, key="history_emotion_filter")

        with f_col2:
            history_search = st.text_input("🔍 Search Text in History:", value="", placeholder="Type keyword...", key="hist_search")

        with f_col3:
            min_confidence = st.slider("Filter Min Confidence %:", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="hist_conf_slider")

        filtered_hist = hist_df.copy()
        if history_emotion_filter != "All Emotions":
            filtered_hist = filtered_hist[filtered_hist["Predicted Emotion"] == history_emotion_filter]

        if history_search.strip() != "":
            filtered_hist = filtered_hist[filtered_hist["Text"].str.contains(history_search, case=False, na=False)]

        filtered_hist = filtered_hist[filtered_hist["Confidence %"] >= min_confidence]

        st.dataframe(filtered_hist, use_container_width=True)

        report_html = generate_html_report(filtered_hist)
        st.download_button(
            "📄 Export History Executive Analytics Report (HTML/PDF)",
            data=report_html.encode("utf-8"),
            file_name="khushi_emotion_history_report.html",
            mime="text/html",
            key="hist_report_btn"
        )

# ----------------------------------------------------------------------------
# Tab 2: Batch Predict & PDF Document Analyzer
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown("#### 📁 Bulk Dataset (CSV) & Document Analyzer (PDF / TXT)")
    st.caption("Upload CSV datasets or full PDF / TXT documents for automated paragraph-by-paragraph emotion analysis.")

    uploaded_file = st.file_uploader("Choose a file (CSV, PDF, TXT):", type=["csv", "pdf", "txt"], key="batch_file_uploader")

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext == "pdf":
            st.info("📄 PDF Document detected. Extracting paragraphs for document emotion timeline...")
            paragraphs, pdf_err = extract_text_from_pdf(uploaded_file)
            if pdf_err or not paragraphs:
                st.error(pdf_err or "No readable text paragraphs found in this PDF.")
                batch_df = None
            else:
                st.success(f"Extracted {len(paragraphs)} text paragraphs from PDF document!")
                batch_df = pd.DataFrame({
                    "Paragraph_ID": [f"Paragraph {i+1}" for i in range(len(paragraphs))],
                    "Text": paragraphs
                })
                text_col = "Text"
        elif file_ext == "txt":
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            paragraphs = [p.strip() for p in raw_text.split('\n') if len(p.strip()) > 10]
            st.success(f"Extracted {len(paragraphs)} lines/paragraphs from TXT document!")
            batch_df = pd.DataFrame({
                "Paragraph_ID": [f"Line {i+1}" for i in range(len(paragraphs))],
                "Text": paragraphs
            })
            text_col = "Text"
        else:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read this CSV file: {e}")
                batch_df = None

        if batch_df is not None:
            st.write("**Data Preview:**", batch_df.head())
            if file_ext == "csv":
                text_col = st.selectbox("Select the column containing text:", options=batch_df.columns)

            if st.button("Run Batch Prediction"):
                with st.spinner(f"Predicting emotions for {len(batch_df)} rows/paragraphs..."):
                    cleaned = batch_df[text_col].astype(str).apply(clean_text)
                    vecs = vectorizer.transform(cleaned)
                    preds = model.predict(vecs)
                    probs = model.predict_proba(vecs)

                    result_df = batch_df.copy()
                    result_df["Text"] = batch_df[text_col]
                    result_df["Predicted Emotion"] = [emotion_mapping[p] for p in preds]
                    result_df["Confidence %"] = [round(probs[i][p]*100, 1) for i, p in enumerate(preds)]

                st.session_state["batch_result"] = result_df

            if "batch_result" in st.session_state:
                result_df = st.session_state["batch_result"]
                st.success(f"Batch prediction completed for {len(result_df)} rows/paragraphs!")

                st.markdown("##### 🎛️ Batch Result Filters")
                b_col1, b_col2, b_col3 = st.columns([1.5, 1.5, 1.5])

                with b_col1:
                    all_emotions = sorted(result_df["Predicted Emotion"].unique().tolist())
                    emo_filter = st.multiselect(
                        "Filter Emotions:",
                        options=all_emotions,
                        default=all_emotions,
                        key="batch_emo_multiselect"
                    )

                with b_col2:
                    batch_search = st.text_input("🔍 Search Batch Text:", value="", placeholder="Keyword...", key="batch_text_search")

                with b_col3:
                    batch_min_conf = st.slider("Min Confidence %:", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="batch_conf_slider")

                filtered_df = result_df.copy()
                if emo_filter:
                    filtered_df = filtered_df[filtered_df["Predicted Emotion"].isin(emo_filter)]

                if batch_search.strip() != "":
                    filtered_df = filtered_df[filtered_df[text_col].astype(str).str.contains(batch_search, case=False, na=False)]

                filtered_df = filtered_df[filtered_df["Confidence %"] >= batch_min_conf]

                st.markdown(f"**Showing {len(filtered_df)} of {len(result_df)} rows:**")
                st.dataframe(filtered_df, use_container_width=True)

                col_dl1, col_dl2 = st.columns([1, 1])

                with col_dl1:
                    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Filtered Results as CSV",
                        data=csv_bytes,
                        file_name="emotion_predictions.csv",
                        mime="text/csv",
                    )

                with col_dl2:
                    batch_html_report = generate_html_report(filtered_df)
                    st.download_button(
                        "📄 Export Executive Analytics Report (HTML/PDF)",
                        data=batch_html_report.encode("utf-8"),
                        file_name="batch_emotion_executive_report.html",
                        mime="text/html",
                        key="batch_report_btn"
                    )

                st.markdown("##### 📊 Emotion Distribution (Filtered Results)")
                dist = filtered_df["Predicted Emotion"].value_counts()
                fig_dist = create_distribution_plotly_chart(dist)
                st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

# ----------------------------------------------------------------------------
# Tab 3: Insights & Model Evaluation
# ----------------------------------------------------------------------------
with tab_insights:
    st.markdown("#### 📊 3-Model Comprehensive Comparison Chart")
    if scores and "accuracy" in scores:
        c1, c2 = st.columns([1.5, 1])

        with c1:
            st.markdown("**Accuracy Comparison Across All 3 Model Architectures:**")
            comp_df = pd.DataFrame(
                list(scores["accuracy"].items()), columns=["Model Architecture", "Accuracy Score"]
            )
            comp_df["Accuracy %"] = comp_df["Accuracy Score"] * 100

            fig_comp, ax_comp = plt.subplots(figsize=(6, 3.5))
            fig_comp.patch.set_facecolor('#141824')
            ax_comp.set_facecolor('#141824')

            colors = ['#334155', '#475569', '#6366f1']
            bars = ax_comp.bar(comp_df["Model Architecture"], comp_df["Accuracy %"], color=colors, edgecolor='#818cf8', width=0.5)

            for bar in bars:
                height = bar.get_height()
                ax_comp.text(bar.get_x() + bar.get_width()/2., height + 1.2, f'{height:.1f}%', ha='center', va='bottom', color='#ffffff', fontweight='bold')

            ax_comp.set_ylim(0, 105)
            ax_comp.set_ylabel("Accuracy %", color="#ffffff", fontweight="bold")
            ax_comp.spines['top'].set_visible(False)
            ax_comp.spines['right'].set_visible(False)
            ax_comp.spines['left'].set_color('#282e42')
            ax_comp.spines['bottom'].set_color('#282e42')
            ax_comp.tick_params(colors="#cbd5e1", labelsize=8.5)
            plt.xticks(rotation=10)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_comp)

        with c2:
            st.markdown("**Model Benchmark Summary Table**")
            acc_table = pd.DataFrame(
                list(scores["accuracy"].items()), columns=["Model Architecture", "Accuracy Score"]
            ).set_index("Model Architecture")
            acc_table["Accuracy %"] = acc_table["Accuracy Score"].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(acc_table[["Accuracy %"]], use_container_width=True)

            st.markdown("""
            <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid #6366f1; border-radius: 10px; padding: 12px;">
                <strong style="color: #38bdf8;">🏆 Key Takeaway:</strong><br/>
                Logistic Regression + TF-IDF outperformed Naive Bayes (BoW) by <strong>+9.47%</strong> and Naive Bayes (TF-IDF) by <strong>+20.19%</strong>!
            </div>
            """, unsafe_allow_html=True)

    # Confusion Matrix & Classification Metrics Sub-section
    if scores and "confusion_matrix" in scores and "class_report" in scores:
        st.markdown("---")
        st.markdown("#### 🎯 Confusion Matrix & Class Performance Heatmap")
        cm_col, rep_col = st.columns([1.2, 1.2])

        with cm_col:
            st.markdown("**Confusion Matrix (Count Heatmap)**")
            cm_data = scores["confusion_matrix"]
            labels = ["Sadness", "Anger", "Love", "Surprise", "Fear", "Joy"]
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            fig_cm.patch.set_facecolor('#141824')
            ax_cm.set_facecolor('#141824')

            sns.heatmap(cm_data, annot=True, fmt="d", cmap="magma", xticklabels=labels, yticklabels=labels, ax=ax_cm, cbar=False)
            ax_cm.set_xlabel("Predicted Label", color="#ffffff", fontweight="bold")
            ax_cm.set_ylabel("True Label", color="#ffffff", fontweight="bold")
            ax_cm.tick_params(colors="#cbd5e1")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_cm)

        with rep_col:
            st.markdown("**Class Precision, Recall & F1-Score Table**")
            class_report_dict = scores["class_report"]
            report_rows = []
            for emo in ["Sadness 😔", "Anger 😡", "Love ❤️", "Surprise 😲", "Fear 😨", "Joy 😊"]:
                if emo in class_report_dict:
                    report_rows.append({
                        "Emotion Class": emo,
                        "Precision": f"{class_report_dict[emo]['precision']*100:.1f}%",
                        "Recall": f"{class_report_dict[emo]['recall']*100:.1f}%",
                        "F1-Score": f"{class_report_dict[emo]['f1-score']*100:.1f}%",
                        "Support": class_report_dict[emo]['support']
                    })
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔤 Word Importance & Keyword Filters")

    feature_names = vectorizer.get_feature_names_out()
    i_col1, i_col2, i_col3, i_col4 = st.columns([1.2, 1.2, 1.5, 1.2])

    with i_col1:
        chosen_emotion = st.selectbox(
            "Select Emotion:",
            options=list(emotion_mapping.values()),
            key="emotion_select_insights",
        )

    with i_col2:
        top_n = st.slider("Top Words Count:", min_value=10, max_value=50, value=20, step=5, key="top_n_slider")

    with i_col3:
        polarity_filter = st.radio(
            "Weight Polarity:",
            options=["Positive (+)", "Negative (-)", "All Weights"],
            horizontal=True,
            key="polarity_radio"
        )

    with i_col4:
        word_search = st.text_input("🔍 Search Word:", value="", placeholder="Search...", key="insights_word_search")

    emotion_idx = [k for k, v in emotion_mapping.items() if v == chosen_emotion][0]
    coefs = model.coef_[emotion_idx]

    feat_df = pd.DataFrame({"Word": feature_names, "Weight": coefs})

    if polarity_filter == "Positive (+)":
        feat_df = feat_df[feat_df["Weight"] > 0].sort_values(by="Weight", ascending=False)
    elif polarity_filter == "Negative (-)":
        feat_df = feat_df[feat_df["Weight"] < 0].sort_values(by="Weight", ascending=True)
    else:
        feat_df["AbsWeight"] = feat_df["Weight"].abs()
        feat_df = feat_df.sort_values(by="AbsWeight", ascending=False).drop(columns=["AbsWeight"])

    if word_search.strip() != "":
        feat_df = feat_df[feat_df["Word"].str.contains(word_search, case=False, na=False)]

    feat_df_top = feat_df.head(top_n)

    top_words = feat_df_top["Word"].tolist()
    top_weights = feat_df_top["Weight"].tolist()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**☁️ Interactive Word Cloud**")
        if WORDCLOUD_AVAILABLE and len(top_words) > 0:
            freq_dict = {w: max(abs(float(wt)), 0.001) for w, wt in zip(top_words, top_weights)}

            wc = WordCloud(
                width=600, height=400,
                background_color="#141824",
                colormap="plasma" if polarity_filter != "Negative (-)" else "cool",
                max_words=top_n,
            ).generate_from_frequencies(freq_dict)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#141824')
            ax.set_facecolor('#141824')
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig)
        elif len(top_words) == 0:
            st.warning("No words match your search filter.")
        else:
            st.info("Install `wordcloud` (`pip install wordcloud`) to visualize word clouds.")

    with col_b:
        st.markdown(f"**📋 Top {len(top_words)} Words Table**")
        top_words_table = pd.DataFrame({
            "Rank": list(range(1, len(top_words) + 1)),
            "Word": top_words,
            "Importance Weight": [f"{wt:+.4f}" for wt in top_weights]
        })
        st.dataframe(top_words_table, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Tab 4: About & Developer Card & Architecture Flowchart
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown("""
    <!-- Developer Profile Card -->
    <div class="theory-card" style="border-left: 6px solid #38bdf8; background: linear-gradient(135deg, rgba(22, 27, 39, 0.95) 0%, rgba(17, 24, 39, 0.95) 100%);">
        <h3 style="color: #cbd5e1 !important; font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">👩‍💻 Project Developer Profile</h3>
        <div style="display: flex; align-items: center; gap: 20px; margin-top: 10px;">
            <div style="background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%); border-radius: 50%; width: 70px; height: 70px; display: flex; align-items: center; justify-content: center; font-size: 2.4rem; box-shadow: 0 0 20px rgba(56, 189, 248, 0.5);">👩‍🔬</div>
            <div>
                <h2 style="background: linear-gradient(135deg, #38bdf8 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem !important; font-weight: 900 !important; margin: 0; padding: 0;">Khushi Singh</h2>
                <p style="color: #38bdf8 !important; margin: 4px 0 10px 0; font-weight: 700; font-size: 1.1rem;">Lead NLP & Machine Learning Engineer</p>
                <p style="color: #e2e8f0 !important; margin: 0; font-size: 1.05rem; line-height: 1.65;">
                    Designed and engineered this end-to-end multi-class emotion detection platform comparing feature extraction techniques 
                    (Bag-of-Words vs. TF-IDF) across Naive Bayes and Logistic Regression machine learning classifiers.
                </p>
            </div>
        </div>
    </div>

    <!-- Card 1: Overview -->
    <div class="theory-card" style="border-left: 6px solid #6366f1;">
        <h3 style="color: #38bdf8 !important; font-size: 1.25rem;">ℹ️ Project Overview</h3>
        <p>This application implements an end-to-end Machine Learning NLP platform that classifies text into 
        6 distinct emotional categories (<strong>Sadness 😔, Anger 😡, Love ❤️, Surprise 😲, Fear 😨, Joy 😊</strong>).</p>
    </div>

    <!-- Card 2: Pipeline Steps -->
    <div class="theory-card" style="border-left: 6px solid #a855f7;">
        <h3 style="color: #c084fc !important; font-size: 1.25rem;">🔄 Step-by-Step Data Processing</h3>
        <ul style="line-height: 1.8;">
            <li><strong>1. Text Cleaning & Translation:</strong> Supports English and Hinglish/Hindi text translation, lowercasing, punctuation/digit removal, non-ASCII emoji filtering, and English stopword stripping (<code>nltk.corpus.stopwords</code>).</li>
            <li><strong>2. TF-IDF Feature Extraction:</strong> Transforms clean sentences into numerical feature vectors. TF-IDF highlights unique emotional keywords while discounting ubiquitous words.</li>
            <li><strong>3. Multi-Class Logistic Regression:</strong> Uses a linear decision boundary to compute class probabilities across all 6 target emotions.</li>
        </ul>
    </div>

    <!-- Card 3: Model Benchmark -->
    <div class="theory-card" style="border-left: 6px solid #4ade80;">
        <h3 style="color: #4ade80 !important; font-size: 1.25rem;">🎯 Why Logistic Regression + TF-IDF?</h3>
        <ul style="line-height: 1.8;">
            <li><strong>High Accuracy:</strong> Achieved <strong>86.28% test accuracy</strong>, outperforming Naive Bayes (Bag of Words: 76.81%, TF-IDF: 66.09%).</li>
            <li><strong>Linear Separability:</strong> Handles high-dimensional sparse TF-IDF vectors exceptionally well.</li>
            <li><strong>Model Interpretability:</strong> Log-odds weights provide direct, readable insights into exact words driving each emotion.</li>
        </ul>
    </div>

    <!-- Card 4: Platform Features -->
    <div class="theory-card" style="border-left: 6px solid #fbbf24;">
        <h3 style="color: #fbbf24 !important; font-size: 1.25rem;">✨ Key Platform Capabilities</h3>
        <ul style="line-height: 1.8;">
            <li><strong>Single Text Prediction & Hinglish Support:</strong> Live prediction, confidence chart, preprocessing inspector, Hinglish translation, AI positive rephraser, emotion-reactive UI themes, and filterable history.</li>
            <li><strong>Batch Processing:</strong> Bulk CSV processing, text column selector, multi-emotion filter, keyword search filter, min confidence threshold slider, downloadable CSV, and HTML executive analytics report export.</li>
    </div>
    """, unsafe_allow_html=True)

    # Card 5: Pipeline Flowchart Image Frame
    st.markdown("""
    <div class="theory-card" style="border-left: 6px solid #38bdf8;">
        <h3 style="color: #38bdf8 !important; font-size: 1.3rem;">📊 Complete NLP Architecture Pipeline Flowchart</h3>
        <p style="color: #cbd5e1 !important; margin-bottom: 8px;">Visual step-by-step workflow of how input text flows from raw input to ML classification:</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image("nlp_pipeline_flowchart.png", use_container_width=True)
    except Exception as e:
        st.info("Flowchart diagram loading...")