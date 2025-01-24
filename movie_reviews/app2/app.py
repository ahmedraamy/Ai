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
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load models
from tensorflow.keras.models import load_model

try:
    # Load the Keras model using TensorFlow's `load_model` method
    tf = pickle.load(open('movie_reviews/app2/artifacts/tf.pkl', 'rb'))  # Update the path as needed
    st.success(" model loaded successfully!")
    time.sleep(1)  # Wait for 1 second
    st.empty()
except Exception as e:
    st.error(f"An error occurred while loading the TensorFlow model: {e}")
    tf_model = None

# Load other models (pickle)
try:
    lr = pickle.load(open('movie_reviews/app2/artifacts/lr.pkl', 'rb'))
    dt = pickle.load(open('movie_reviews/app2/artifacts/dt.pkl', 'rb'))
    svc = pickle.load(open('movie_reviews/app2/artifacts/svc.pkl', 'rb'))
    st.success("models loaded successfully!")
    time.sleep(1)  # Wait for 1 second
    st.empty()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    lr, dt, svc = None, None, None
except Exception as e:
    st.error(f"An error occurred while loading other models: {e}")
    lr, dt, svc = None, None, None

# Initialize stopwords and other NLP tools
try:
    stop_words = stopwords.words('english')
    if 'not' in stop_words:
        stop_words.remove('not')
    if 'no' in stop_words:
        stop_words.remove('no')
except Exception as e:
    st.error(f"Error initializing stopwords: {e}")
    stop_words = []

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Text Preprocessing function
def text_preprocessing(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

# Predict Sentiment function
def predict_feedback(text):
    processed_text = text_preprocessing(text)
    sentiment_score = analyzer.polarity_scores(processed_text)

    if sentiment_score['compound'] > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    
    # Display result using Streamlit markdown
    st.markdown(
        f"<p style='color: red; font-weight: bold;'>The review is <b>{sentiment}</b></p>",
        unsafe_allow_html=True,
    )

# Main App Interface
st.title('Movie Reviews App')

# Text area for user input
user_input = st.text_area("Enter the text for movie review:", "I don't like this product.")

# Button to trigger sentiment prediction
if st.button('Predict Sentiment'):
    if tf or lr or dt or svc:
        prediction = predict_feedback(user_input)
    else:
        st.error("Models could not be loaded. Please check the configuration.")

# Custom Styling for Streamlit app
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

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #1E90FF; font-size: 12px;'>
        <p>Created by Ahmed Ramy</p>
    </div>
    """,
    unsafe_allow_html=True,
)
