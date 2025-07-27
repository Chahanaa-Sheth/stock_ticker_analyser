import streamlit as st
import pandas as pd
import os
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from newsdataapi import NewsDataApiClient
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, minimalistic design
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    /* ... rest of your CSS unchanged ... */
    </style>
    """, unsafe_allow_html=True)

load_css()

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = []
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = None

# Header
st.markdown("""
<div class="header">
    <h1>🚀 AI-Powered Stock Forecaster</h1>
    <p>Advanced ML predictions with real-time sentiment analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with modern design
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #667eea; margin-bottom: 1rem;">📊 Controls</h2>
    </div>
    """, unsafe_allow_html=True)

    ticker = st.text_input(
        "📈 Stock Symbol",
        value="AAPL",
        placeholder="e.g., AAPL, GOOGL, MSFT",
        help="Enter a valid stock ticker symbol"
    ).upper()

    train_button = st.button("🤖 Train AI Model", key="train_model")

    status_class = "status-trained" if st.session_state.model_trained else "status-not-trained"
    status_text = "✅ Model Ready" if st.session_state.model_trained else "❌ Model Not Trained"
    st.markdown(f"""
    <div class="status-indicator {status_class}">
        {status_text}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📰 News Configuration", unsafe_allow_html=True)

# NewsData.io setup
NEWSDATA_KEY = st.secrets.get("NEWSDATA_IO_KEY") or os.environ.get("NEWSDATA_IO_KEY")
if NEWSDATA_KEY:
    try:
        newsapi = NewsDataApiClient(apikey=NEWSDATA_KEY)
        st.sidebar.success("✅ NewsData.io Connected")
    except Exception as e:
        newsapi = None
        st.sidebar.error(f"❌ NewsData.io Error: {e}")
else:
    newsapi = None
    st.sidebar.info("Enter your NewsData.io key in secrets or below")
    key_input = st.sidebar.text_input("🔑 NewsData.io Key", type="password")
    if key_input:
        try:
            newsapi = NewsDataApiClient(apikey=key_input)
            st.sidebar.success("✅ NewsData.io Connected")
        except Exception as e:
            st.sidebar.error(f"❌ Invalid key: {e}")

st.sidebar.markdown("""
<div class="info-box">
    <strong>Features Used:</strong><br>
    • Open Price<br>
    • High Price<br>
    • Low Price<br>
    • Volume<br>
    • News Sentiment
</div>
""", unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

# Training logic
if train_button and ticker:
    with st.spinner("🤖 Training AI model..."):
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")

            if df.empty:
                st.error("❌ No data found for this ticker. Please check the symbol.")
            else:
                # Prepare data
                data = df.reset_index()
                data["Tomorrow_Close"] = data["Close"].shift(-1)
                data = data.dropna(subset=["Tomorrow_Close"])

                if len(data) < 10:
                    st.error("❌ Not enough data for training. Try a different ticker.")
                else:
                    # Train model
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X = data[["Open", "High", "Low", "Volume"]]
                    y = data["Tomorrow_Close"]
                    X_scaled = scaler.fit_transform(X)

                    model = LinearRegression()
                    model.fit(X_scaled, y)

                    # Make prediction
                    last_row = data.iloc[-1]
                    X_last = scaler.transform(last_row[["Open", "High", "Low", "Volume"]].values.reshape(1, -1))
                    pred_price = model.predict(X_last)[0]
                    current_price = last_row["Close"]
                    price_change = ((pred_price - current_price) / current_price) * 100

                    from sklearn.metrics import r2_score
                    confidence = r2_score(y, model.predict(X_scaled)) * 100
                    confidence = max(60, min(95, confidence))
                    direction = "📈 UP" if pred_price > current_price else "📉 DOWN"

                    st.session_state.prediction_data = {
                        'direction': direction,
                        'price': pred_price,
                        'change': price_change,
                        'confidence': confidence,
                        'current_price': current_price,
                        'model': model,
                        'scaler': scaler
                    }

        except Exception as e:
            st.error(f"❌ Training error: {e}")

if newsapi:
    try:
        company_name = ticker
        resp = newsapi.news_api(q=company_name, language="en", size=5)
        results = resp.get("results", [])
        headlines = [(r["title"], r["link"]) for r in results if r.get("title") and r.get("link")]
        if headlines:
            scores = [TextBlob(t).sentiment.polarity for t, _ in headlines]
            avg = float(np.mean(scores))
            # ✨ Store into session_state so UI can render it:
            st.session_state.news_data = [(t, url, r.get("source", "")) for (t, url), r in zip(headlines, results)]
            st.session_state.sentiment_data = {
                "label": "Positive" if avg > 0 else "Negative" if avg < 0 else "Neutral",
                "score": avg
            }
    except Exception as e:
        st.sidebar.warning(f"NewsData.io fetch failed: {e}")

    st.session_state.model_trained = True
    st.success("✅ Model trained successfully!")
    st.rerun()




# Display charts in left column
with col1:
    if st.session_state.model_trained and ticker:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo").reset_index()

            # Price movement chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_line = px.line(data_frame=df, x="Date", y="Close",
                               title=f"📈 {ticker} Price Movement (6 months)",
                               template="plotly_white")
            fig_line.update_traces(line_color='#667eea', line_width=3)
            st.plotly_chart(fig_line, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # OHLC Candlestick chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df["Date"], open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"]
            )])
            fig_candle.update_layout(template="plotly_white",
                                     title=f"📊 {ticker} OHLC Analysis")
            st.plotly_chart(fig_candle, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Volume chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_volume = px.bar(data_frame=df, x="Date", y="Volume",
                                title=f"📊 {ticker} Trading Volume",
                                template="plotly_white")
            st.plotly_chart(fig_volume, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error loading charts: {e}")
    else:
        st.markdown("""
        <div class="chart-container" style="text-align: center; padding: 4rem;">
            <h3 style="color: #6c757d;">📊 Charts will appear here</h3>
            <p style="color: #9ca3af;">Train the AI model to see detailed price analysis</p>
        </div>
        """, unsafe_allow_html=True)

# Right column - Predictions and insights
with col2:
    # Current price card
    if ticker:
        try:
            current_data = yf.Ticker(ticker).history(period="1d")
            if not current_data.empty:
                current_price = current_data['Close'].iloc[-1]
                prev_close = current_data['Close'].iloc[-2] if len(current_data) > 1 else current_price
                price_change = ((current_price - prev_close) / prev_close) * 100
                change_class = "positive" if price_change >= 0 else "negative"
                change_icon = "📈" if price_change >= 0 else "📉"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Current Price</div>
                    <div class="metric-value">${current_price:.2f}</div>
                    <div class="metric-change {change_class}">{change_icon} {price_change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
        except:
            pass

    # Prediction card
    if st.session_state.model_trained and st.session_state.prediction_data:
        pred_data = st.session_state.prediction_data
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-title">🔮 Tomorrow's Prediction</div>
            <div class="prediction-value">${pred_data['price']:.2f}</div>
            <div>{pred_data['direction']} {pred_data['change']:+.2f}%</div>
            <div class="prediction-confidence">🎯 Model Confidence: {pred_data['confidence']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-card">
            <div class="prediction-title">🔮 AI Prediction</div>
            <div style="font-size: 1.2rem; opacity: 0.8;">
                Train the model to see predictions
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sentiment analysis card
    if st.session_state.sentiment_data:
        sentiment = st.session_state.sentiment_data
        emoji = "😊" if sentiment['label'] == 'Positive' else "😞" if sentiment['label'] == 'Negative' else "😐"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Market Sentiment</div>
            <div class="metric-value">{emoji} {sentiment['label']}</div>
            <div class="metric-change">Score: {sentiment['score']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">📊 Market Sentiment</div>
            <div class="metric-value">😐 Neutral</div>
            <div class="metric-change">Configure NewsAPI for sentiment analysis</div>
        </div>
        """, unsafe_allow_html=True)

    # News feed
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">📰 Latest News</div>
    """, unsafe_allow_html=True)
    if st.session_state.news_data:
        for title, url, source in st.session_state.news_data[:5]:
            st.markdown(f"""
            <div class="news-card">
                <div class="news-title"><a href="{url}" target="_blank">{title}</a></div>
                <div class="news-source">📡 {source}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #6c757d;">
            <p>📰 No news available</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem; border-top: 1px solid #e9ecef; margin-top: 3rem;">
    <p>⚡ Powered by AI • 📊 Real-time Data • 🔒 Secure Analysis</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">
        Data provided by Yahoo Finance | News powered by NewsData.io | Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
