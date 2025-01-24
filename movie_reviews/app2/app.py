import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained models (if applicable)
try:
    lr = pickle.load(open('movie_reviews/app2/artifacts/lr.pkl', 'rb'))
    tf = pickle.load(open('movie_reviews/app2/artifacts/tf.pkl', 'rb'))
    dt = pickle.load(open('movie_reviews/app2/artifacts/dt.pkl', 'rb'))
    svc = pickle.load(open('movie_reviews/app2/artifacts/svc.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")
    st.stop()

stop_words = stopwords.words('english')
stop_words.remove('not')
stop_words.remove('no')

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def text_preprocessing(text):
    # Preprocess the text for tokenization and lemmatization
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

def predict_feedback(text):
    # Predict sentiment using Vader Sentiment Analyzer
    processed_text = text_preprocessing(text)
    sentiment_score = analyzer.polarity_scores(processed_text)

    if sentiment_score['compound'] > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    st.markdown(
        f"<p style='color: red; font-weight: bold;'>The review is <b>{sentiment}</b></p>",
        unsafe_allow_html=True,
    )

def predict_movie_sentiment(text):
    # Example prediction function using a model (e.g., lr, tf, dt, or svc)
    processed_text = text_preprocessing(text)
    
    # Assuming you want to use the logistic regression model as an example
    text_vectorized = tf.transform([processed_text])  # Transform the text using tfidf
    prediction = lr.predict(text_vectorized)  # Predict sentiment using the model
    sentiment = 'Positive' if prediction == 1 else 'Negative'

    st.markdown(
        f"<p style='color: green; font-weight: bold;'>The review is <b>{sentiment}</b></p>",
        unsafe_allow_html=True,
    )

# Streamlit UI
st.title('Movie Reviews App')

user_input = st.text_area("Enter the text for movie review:", "I don't like this product.")

if st.button('Predict Sentiment'):
    # Using Vader sentiment analysis for the user input
    predict_feedback(user_input)
    # Alternatively, you can call the movie sentiment prediction function like so:
    # predict_movie_sentiment(user_input)

# Custom styling
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
