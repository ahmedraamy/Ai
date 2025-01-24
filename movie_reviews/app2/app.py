import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keras.models import load_model

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load pre-trained models and setup pre-processing
try:
    # Load the TensorFlow/Keras model (update path if necessary)
    tf = pickle.load(open('movie_reviews/app2/artifacts/tf.pkl', 'rb'))

    # Load other models
    lr = pickle.load(open('movie_reviews/app2/artifacts/lr.pkl', 'rb'))
    dt = pickle.load(open('movie_reviews/app2/artifacts/dt.pkl', 'rb'))
    svc = pickle.load(open('movie_reviews/app2/artifacts/svc.pkl', 'rb'))

    # Pre-processing setup
    stop_words = stopwords.words('english')
    if 'not' in stop_words:
        stop_words.remove('not')
    if 'no' in stop_words:
        stop_words.remove('no')

    lemmatizer = WordNetLemmatizer()
    analyzer = SentimentIntensityAnalyzer()

except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")
    st.stop()

# Pre-process the input text
def text_preprocessing(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetical characters
    text = word_tokenize(text)  # Tokenize text into words
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize words
    text = ' '.join(text)  # Join words back into a string
    return text

# Sentiment prediction function
def predict_feedback(text):
    processed_text = text_preprocessing(text)  # Preprocess the text
    sentiment_score = analyzer.polarity_scores(processed_text)  # Get sentiment scores

    if sentiment_score['compound'] > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    # Display the sentiment result and score
    st.markdown(
        f"<p style='color: red; font-weight: bold;'>The review is <b>{sentiment}</b></p>",
        unsafe_allow_html=True,
    )
    st.write(f"Sentiment Score: {sentiment_score}")

# App Layout and Functionality
st.title('Movie Reviews Sentiment Analysis')

# Input text area
user_input = st.text_area("Enter the text for movie review:", "I don't like this product.")

if st.button('Predict Sentiment'):
    if not user_input.strip():
        st.warning("Please enter a valid review.")
    else:
        predict_feedback(user_input)

# Custom Styling for the App
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
