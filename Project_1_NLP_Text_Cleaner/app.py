import re
import string
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def load_sms(uploaded_file=None):
    """Supports Kaggle spam.csv, standard SMS CSV, or the public SMS TSV mirror."""
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, sep='\t', header=None, names=['label', 'message'])
        else:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
    else:
        local = Path(__file__).with_name('spam.csv')
        local_tsv = Path(__file__).with_name('sms.tsv')
        if local.exists():
            df = pd.read_csv(local, encoding='latin-1')
        elif local_tsv.exists():
            df = pd.read_csv(local_tsv, sep='\t', header=None, names=['label', 'message'])
        else:
            df = pd.read_csv(DATA_URL, sep='\t', header=None, names=['label','message'])
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'v1' in df and 'v2' in df:
        df = df.rename(columns={'v1':'label', 'v2':'message'})
    if 'label' not in df or 'message' not in df:
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]:'label', df.columns[1]:'message'})
        else:
            raise ValueError('CSV needs label and message columns (or v1 and v2).')
    df = df[['label','message']].dropna().copy()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['message'] = df['message'].astype(str)
    df = df[df['label'].isin(['spam','ham'])]
    if df.empty or df['label'].nunique() < 2:
        raise ValueError('Dataset must contain both ham and spam labels.')
    return df.reset_index(drop=True)


def apply_style():
    st.markdown("""
    <style>
      .block-container {max-width: 1100px; padding-top: 2.6rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.025em;}
      h1 {font-size: 2.15rem !important; font-weight: 700 !important; margin-bottom: 0.2rem !important;}
      h3 {font-size: 1.10rem !important; font-weight: 650 !important;}
      [data-testid="stMetric"] {background: rgba(128, 128, 128, .045); border: 1px solid rgba(128,128,128,.17); border-radius: 9px; padding: 13px 16px;}
      [data-testid="stMetricLabel"] {font-size: .85rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 650;}
      div[data-testid="stButton"] button[kind="primary"] {border-radius: 7px; font-weight: 600;}
      div[data-testid="stDataFrame"] {border-radius: 8px; overflow: hidden;}
      .eyebrow {font-size: .75rem; color: #64748b; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .65rem;}
      .subtle {font-size: .89rem; color: #64748b; margin-bottom: 1.5rem;}
      .footer-rule {border-top: 1px solid rgba(128,128,128,.2); margin-top: 1.9rem; padding-top: .8rem; color:#64748b; font-size:.8rem;}
    </style>
    """, unsafe_allow_html=True)

def intro(project, title, subtitle):
    st.markdown(f'<div class="eyebrow">{project}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="subtle">{subtitle}</p>', unsafe_allow_html=True)


st.set_page_config(page_title="Text analysis | NLP", page_icon="◧", layout="wide")
apply_style()
intro("NLP / MINI PROJECT 01", "Text Cleaner & Keyword Finder", "Explore the patterns behind a collection of SMS messages.")
with st.sidebar:
    st.header("Dataset settings")
    upload = st.file_uploader("Upload a CSV or TSV", type=["csv", "tsv"], help="Optional. Kaggle spam.csv is supported.")
    number = st.slider("Messages to analyze", 100, 200, 150, 10)
    st.caption("Leave the upload empty to use the public SMS dataset.")

stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)

def clean(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join(stemmer.stem(word) for word in text.split() if word not in stop_words and len(word) > 1)

try:
    df = load_sms(upload)
    sample = df.sample(n=min(number, len(df)), random_state=42).copy()
    sample["cleaned"] = sample["message"].map(clean)
    terms = [word for entry in sample["cleaned"] for word in entry.split()]
    common = Counter(terms).most_common(10)
    nonempty = sample.loc[sample["cleaned"].str.strip().ne(""), "cleaned"]
    if nonempty.empty:
        st.warning("No valid words were found after cleaning.")
        st.stop()
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(nonempty)
    keywords = (pd.Series(matrix.sum(axis=0).A1, index=vectorizer.get_feature_names_out())
                .nlargest(10).rename_axis("Keyword").reset_index(name="TF-IDF score"))

    a, b, c = st.columns(3)
    a.metric("Messages analyzed", f"{len(sample):,}")
    b.metric("Tokens after cleaning", f"{len(terms):,}")
    c.metric("Unique terms", f"{len(set(terms)):,}")
    st.divider()
    tab1, tab2 = st.tabs(["Word analysis", "Cleaned messages"])
    with tab1:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Most frequent terms")
            st.caption("Top 10 stemmed words across the selected messages")
            if common:
                labels, values = zip(*reversed(common))
                fig, ax = plt.subplots(figsize=(6.2, 4.3))
                fig.patch.set_alpha(0)
                ax.set_facecolor('none')
                ax.barh(labels, values, color="#2b5974", height=.64)
                ax.set_xlabel("Occurrences")
                ax.spines[['top','right','left']].set_visible(False)
                ax.grid(axis='x', alpha=.15)
                ax.set_axisbelow(True)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        with right:
            st.subheader("Top TF-IDF keywords")
            st.caption("Ranked by the total TF-IDF score across messages")
            st.dataframe(keywords.style.format({"TF-IDF score": "{:.3f}"}), hide_index=True, use_container_width=True)
        st.info("Processing: lowercase → remove punctuation and numbers → remove stop words → Porter stemming → TF-IDF.", icon="ℹ️")
    with tab2:
        st.subheader("Before and after cleaning")
        st.dataframe(sample[["message", "cleaned"]].rename(columns={"message": "Original message", "cleaned": "Cleaned text"}),
                     hide_index=True, use_container_width=True, height=390)
        st.download_button("Export cleaned CSV", sample[["label", "message", "cleaned"]].to_csv(index=False).encode(),
                           "cleaned_messages.csv", "text/csv")
    st.markdown('<div class="footer-rule">Data source: SMS Spam Collection · Analysis sample is reproducible (random seed 42).</div>', unsafe_allow_html=True)
except Exception as exc:
    st.error(f"Unable to process the dataset: {exc}")
    st.caption("Upload the Kaggle SMS spam.csv file, or save it in the app folder.")
