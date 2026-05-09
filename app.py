
import streamlit as st
from textblob import TextBlob
from langdetect import detect
import plotly.express as px
import pandas as pd
import re

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Ultimate AI Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)

# ---------------- PREPROCESSING ---------------- #

def preprocess(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    return text

# ---------------- LANGUAGE DETECTION ---------------- #

def detect_language(text):

    try:

        lang = detect(text)

        languages = {
            "en": "English 🇺🇸",
            "hi": "Hindi 🇮🇳",
            "te": "Telugu 🇮🇳",
            "ta": "Tamil 🇮🇳",
            "fr": "French 🇫🇷",
            "es": "Spanish 🇪🇸"
        }

        return languages.get(lang, lang)

    except:

        return "Unknown"

# ---------------- SENTIMENT ANALYSIS ---------------- #

def get_sentiment(text):

    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    if polarity > 0:

        return "Positive 😊"

    elif polarity < 0:

        return "Negative 😠"

    else:

        return "Neutral 😐"

# ---------------- EMOTION DETECTION ---------------- #

def detect_emotion(text):

    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    if polarity > 0.5:

        return "Happy 😊"

    elif polarity < -0.5:

        return "Angry 😠"

    else:

        return "Neutral 😐"

# ---------------- AUTO SUMMARIZATION ---------------- #

def summarize_review(text):

    sentences = text.split(".")

    return ".".join(sentences[:2])

# ---------------- KEYWORD HIGHLIGHTING ---------------- #

def highlight_keywords(text):

    suspicious_words = [
        "best",
        "amazing",
        "perfect",
        "incredible",
        "outstanding",
        "must",
        "everyone",
        "excellent",
        "fantastic",
        "unbelievable",
        "worst",
        "poor",
        "bad",
        "cheap",
        "expensive",
        "great"
    ]

    highlighted_text = text

    for word in suspicious_words:

        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)

        highlighted_text = pattern.sub(
            f"<mark style='background-color:yellow; color:black;'>{word}</mark>",
            highlighted_text
        )

    return highlighted_text

# ---------------- EXPLAINABLE AI ---------------- #

def explain_review(text):

    suspicious_words = [
        "best",
        "amazing",
        "perfect",
        "incredible",
        "outstanding",
        "must",
        "everyone",
        "excellent",
        "fantastic",
        "unbelievable"
    ]

    reasons = []

    text_lower = text.lower()

    if text.count("!") > 3:

        reasons.append("Too many exclamation marks")

    words = text_lower.split()

    repeated = len(words) - len(set(words))

    if repeated > 2:

        reasons.append("Repeated words detected")

    found_words = []

    for word in suspicious_words:

        if word in text_lower:

            found_words.append(word)

    if found_words:

        reasons.append(
            "Suspicious promotional words: "
            + ", ".join(found_words)
        )

    if len(words) < 5:

        reasons.append("Review too short")

    if not reasons:

        reasons.append("Review appears natural")

    return reasons

# ---------------- AI GENERATED REVIEW DETECTION ---------------- #

def detect_ai_review(text):

    ai_phrases = [
        "overall experience",
        "highly recommend",
        "in conclusion",
        "exceptional service",
        "valuable experience",
        "wonderful hospitality"
    ]

    text_lower = text.lower()

    score = 0

    for phrase in ai_phrases:

        if phrase in text_lower:

            score += 1

    if score >= 2:

        return "Possibly AI-Generated 🤖"

    else:

        return "Likely Human-Written 👤"

# ---------------- MULTI-MODAL ANALYSIS ---------------- #

def multimodal_analysis(text):

    word_count = len(text.split())

    char_count = len(text)

    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    exclamation_count = text.count("!")

    return {
        "Word Count": word_count,
        "Character Count": char_count,
        "Uppercase Ratio": round(uppercase_ratio, 2),
        "Exclamation Marks": exclamation_count
    }

# ---------------- FAKE REVIEW DETECTION ---------------- #

def predict_review(text):

    suspicious_words = [
        "best",
        "perfect",
        "amazing",
        "must",
        "everyone",
        "excellent"
    ]

    text_lower = text.lower()

    suspicious_score = 0

    for word in suspicious_words:

        if word in text_lower:

            suspicious_score += 1

    if suspicious_score >= 2:

        return "Deceptive Review 🚨"

    else:

        return "Truthful Review ✅"

# ---------------- UI ---------------- #

st.title("🔍 AI-Powered Fake Review Detection System")

st.write("""
An intelligent AI-powered platform that detects fake reviews using NLP, sentiment analysis, explainable AI, multilingual processing, and real-time analytics.
""")

# ---------------- INPUT ---------------- #

user_review = st.text_area("Enter Review")

# ---------------- BUTTON ---------------- #

if st.button("Analyze Review"):

    if user_review.strip() == "":

        st.warning("Please enter a review.")

    else:

        result = predict_review(user_review)

        sentiment = get_sentiment(user_review)

        emotion = detect_emotion(user_review)

        language = detect_language(user_review)

        ai_result = detect_ai_review(user_review)

        reasons = explain_review(user_review)

        multimodal_metrics = multimodal_analysis(user_review)

        summary = summarize_review(user_review)

        highlighted_review = highlight_keywords(user_review)

        # ---------------- OUTPUT ---------------- #

        st.subheader("Prediction")

        if "Truthful" in result:

            st.success(result)

        else:

            st.error(result)

        st.subheader("Detected Language")

        st.info(language)

        st.subheader("Sentiment Analysis")

        st.info(sentiment)

        st.subheader("Emotion Detection")

        st.info(emotion)

        st.subheader("AI Generated Review Detection")

        st.warning(ai_result)

        st.subheader("Auto Summarization")

        st.success(summary)

        st.subheader("Keyword Highlighting")

        st.markdown(
            highlighted_review,
            unsafe_allow_html=True
        )

        st.subheader("Explainable AI Analysis")

        for reason in reasons:

            st.write("•", reason)

        st.subheader("Dashboard Analytics")

        analytics_df = pd.DataFrame({
            "Metric": list(multimodal_metrics.keys()),
            "Value": list(multimodal_metrics.values())
        })

        fig = px.bar(
            analytics_df,
            x="Metric",
            y="Value",
            title="Review Analytics Dashboard"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Multi-Modal AI Metrics")

        for key, value in multimodal_metrics.items():

            st.write(f"**{key}:** {value}")
