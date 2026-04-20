import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sentimen MBG - Analytics", page_icon="🍱", layout="wide")

# Custom CSS untuk tampilan premium dan scannability
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextArea textarea { border-radius: 10px; border: 1px solid #4b5267; }
    .stMetric { border: 1px solid #30363d; padding: 15px; border-radius: 10px; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE HISTORY (Session State) ---
# Fitur ini menyimpan data analisis selama tab browser tidak di-refresh
if 'history' not in st.session_state:
    st.session_state.history = []

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Mengambil model terbaik dari folder notebook
    model = tf.keras.models.load_model('notebook/best_model.keras')
    with open('notebook/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.header("⚙️ Model Dashboard")
    st.markdown("---")
    st.write("**Arsitektur:** Bidirectional GRU")
    st.write("**Tuning:** Optuna (Bayesian)")
    st.write("**Max Length:** 200 Tokens")
    
    if st.button("Reset Market Trend 🗑️", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    st.info("Setiap analisis akan tercatat di grafik tren market di bawah.")

# --- MAIN UI ---
st.title("🍱 Sentiment Intelligence: Makan Bergizi Gratis")
st.markdown("Menganalisis persepsi publik dengan arsitektur **Bi-GRU** yang memproses data secara dua arah.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Data")
    user_input = st.text_area("Masukkan opini publik:", height=150, placeholder="Tulis komentar di sini...")
    analyze = st.button("Jalankan Analisis 🔍", use_container_width=True)

with col2:
    st.subheader("Analytic Visualization")
    if analyze and user_input:
        # 1. Preprocessing (Sesuai dengan image_1481ad.png)
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        
        # 2. Prediction (Sesuai dengan image_1489c8.png)
        raw_score = model.predict(padded)[0][0]
        
        # 3. Mapping (Berdasarkan tes kamu: > 0.5 adalah Negatif)
        prob_neg = raw_score * 100
        prob_pos = (1 - raw_score) * 100
        
        if raw_score > 0.5:
            label, color, emoji = "NEGATIF", "#FF4B4B", "😡"
            display_val = prob_neg
        else:
            label, color, emoji = "POSITIF", "#00C04B", "😊"
            display_val = prob_pos

        # 4. Save to History for Trend Chart
        st.session_state.history.append({
            "Waktu": datetime.now().strftime("%H:%M:%S"),
            "Sentiment": label,
            "Positivity": round(prob_pos, 2),
            "Emoji": emoji
        })

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = display_val,
            title = {'text': f"Confidence: {label}"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [{'range': [0, 100], 'color': "#2c2c2c"}]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Input opini untuk melihat visualisasi probabilitas.")

# --- HASIL DETAIL & TRADING TREND ---
if analyze and user_input:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### Sentimen:<br><span style='color:{color}'>{label} {emoji}</span>", unsafe_allow_html=True)
    with c2:
        st.metric("Bullish Power (Positif)", f"{prob_pos:.1f}%")
    with c3:
        st.metric("Bearish Power (Negatif)", f"{prob_neg:.1f}%")

    # --- FITUR BARU: SENTIMENT MARKET TREND ---
    st.subheader("💹 Sentiment Market Trend (History)")
    if len(st.session_state.history) > 0:
        hist = st.session_state.history
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=[d['Waktu'] for d in hist], 
            y=[d['Positivity'] for d in hist],
            mode='lines+markers',
            name='Positivity Trend',
            line=dict(color='#00C851', width=4),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 81, 0.1)'
        ))
        
        fig_trend.update_layout(
            yaxis=dict(title="Positivity Strength (%)", range=[0, 100], gridcolor="#30363d"),
            xaxis=dict(title="Timestamp", gridcolor="#30363d"),
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Explainable AI (Analogi Detektif)
    with st.expander("🕵️ Lihat Analogi Detektif (Explainable AI)"):
        st.write(f"""
        Model bekerja seperti detektif yang mencari petunjuk dari dua arah (**Bi-GRU**) 
        untuk memastikan apakah opini ini mengandung sentimen positif atau negatif. 
        Keputusan akhir: **{label}** dengan keyakinan **{display_val:.2f}%**.
        """)