import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer  # Add this import for tokenizer

# Load model
model = load_model('model.keras')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')
stop_words.remove('not')
stop_words.remove('no')

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def text_preprocessing(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

def predict_feedback(text):
    processed_text = text_preprocessing(text)
    sentiment_score = analyzer.polarity_scores(processed_text)

    if sentiment_score['compound'] > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    # Call st.markdown only once to display the sentiment analysis result
    st.markdown(
        f"<p style='color: red; font-weight: bold;'>The review is <b>{sentiment}</b></p>",
        unsafe_allow_html=True,
    )

def predict_movie_sentiment(text):
    processed_text = text_preprocessing(text)
    sentiment_score = analyzer.polarity_scores(processed_text)

st.title('Movie Reviews App')

user_input = st.text_area("Enter the text for movie review:" , "I don't like this product.")

if st.button('Predict Sentiment'):
    prediction = predict_feedback(user_input)
    # No need to write prediction here as it's just the function call

st.markdown(
    """
    <style>
    .main { 
        background-color: #f0f2f6; 
    }
    h1 {
        color: #1E90FF; 
    }
    .stButton>button {
        background-color: #1E90FF; 
        color: white;
    }
    .stTextInput>div>input {
        background-color: #ffffff; 
        color: #1E90FF; 
        border: 2px solid #1E90FF; 
        padding: 10px;
    }
    .stTextInput>div>input:focus {
        border-color: #1E90FF; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align: center; color: #1E90FF; font-size: 12px;'>
        <p>Created by Ahmed Ramy</p>
    </div>
    """,
    unsafe_allow_html=True,
)
