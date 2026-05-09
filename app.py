import streamlit as st
from transformers import pipeline
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

# ---------------- LOAD BERT MODEL ---------------- #

@st.cache_resource
def load_model():

    classifier = pipeline(
        "text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    return classifier

classifier = load_model()

# ---------------- LOAD EMOTION MODEL ---------------- #

@st.cache_resource
def load_emotion_model():

    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )

    return emotion_classifier

emotion_classifier = load_emotion_model()

# ---------------- LOAD SUMMARIZATION MODEL ---------------- #

@st.cache_resource
def load_summarizer():

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    return summarizer

summarizer = load_summarizer()

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

    result = emotion_classifier(text)

    emotion = result[0][0]['label']

    score = result[0][0]['score'] * 100

    return f"{emotion} ({score:.2f}% confidence)"

# ---------------- AUTO SUMMARIZATION ---------------- #

def summarize_review(text):

    if len(text.split()) < 20:

        return "Review too short to summarize."

    summary = summarizer(
        text,
        max_length=40,
        min_length=10,
        do_sample=False
    )

    return summary[0]['summary_text']

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
        "unbelievable"
    ]

    highlighted_text = text

    for word in suspicious_words:

        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)

        highlighted_text = pattern.sub(
            f"<span style='color:red; font-weight:bold;'>{word}</span>",
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

# ---------------- MULTI-MODAL AI ---------------- #

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

    cleaned_text = preprocess(text)

    result = classifier(cleaned_text)

    label = result[0]['label']

    score = result[0]['score'] * 100

    suspicious_words = [
        "best",
        "perfect",
        "amazing",
        "must",
        "everyone"
    ]

    suspicious_score = 0

    for word in suspicious_words:

        if word in cleaned_text:

            suspicious_score += 1

    if suspicious_score >= 2 and "5 stars" in label:

        return f"Deceptive Review ({score:.2f}% confidence)"

    else:

        return f"Truthful Review ({score:.2f}% confidence)"

# ---------------- UI ---------------- #

st.title("🔍 Ultimate AI Fake Review Detection System")

st.write(
    """
Transformer-based multilingual fake review detection system
with Emotion Detection, Explainable AI,
Language Detection, Dashboard Analytics,
Keyword Highlighting, Auto Summarization,
and Multi-Modal AI capabilities.
"""
)

# ---------------- INPUT ---------------- #

user_review = st.text_area("Enter Review")

# ---------------- BUTTON ---------------- #

if st.button("Analyze Review"):

    if user_review.strip() == "":

        st.warning("Please enter a review.")

    else:

        # Prediction
        result = predict_review(user_review)

        # Sentiment
        sentiment = get_sentiment(user_review)

        # Emotion
        emotion = detect_emotion(user_review)

        # Language
        language = detect_language(user_review)

        # AI Detection
        ai_result = detect_ai_review(user_review)

        # Explainable AI
        reasons = explain_review(user_review)

        # Metrics
        multimodal_metrics = multimodal_analysis(user_review)

        # Summary
        summary = summarize_review(user_review)

        # Highlighted Text
        highlighted_review = highlight_keywords(user_review)

        # ---------------- OUTPUT ---------------- #

        st.subheader("Prediction")

        if "Truthful" in result:

            st.success(result)

        else:

            st.error(result)

        # Language
        st.subheader("Detected Language")
        st.info(language)

        # Sentiment
        st.subheader("Sentiment Analysis")
        st.info(sentiment)

        # Emotion
        st.subheader("Emotion Detection")
        st.info(emotion)

        # AI Detection
        st.subheader("AI Generated Review Detection")
        st.warning(ai_result)

        # Auto Summary
        st.subheader("Auto Summarization")
        st.success(summary)

        # Keyword Highlighting
        st.subheader("Keyword Highlighting")

        st.markdown(
            highlighted_review,
            unsafe_allow_html=True
        )

        # Explainable AI
        st.subheader("Explainable AI Analysis")

        for reason in reasons:

            st.write("•", reason)

        # Dashboard
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

        # Multi-Modal AI
        st.subheader("Multi-Modal AI Metrics")

        for key, value in multimodal_metrics.items():

            st.write(f"**{key}:** {value}")