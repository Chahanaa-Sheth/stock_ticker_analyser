# sentiment_analysis.py

import requests
from transformers import pipeline

# Load Hugging Face sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def get_news_sentiment_with_headlines(ticker):
    api_key = "e1c2c7865acb46f082f4f1e31338d2f1"  # Use your actual NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}&language=en"

    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
    except Exception as e:
        print(f"News API error: {e}")
        return []

    sentiment_scores = []
    for article in articles[:5]:  # Limit to top 5 news items
        title = article.get("title", "")
        if title:
            result = sentiment_pipeline(title)[0]
            score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
            sentiment_scores.append((title, round(score, 2)))

    return sentiment_scores
