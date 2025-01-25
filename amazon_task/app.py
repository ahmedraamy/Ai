import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load pre-trained models (if applicable)
try:
  lr = pickle.load(open('amazon_task/artifacts/lr.pkl', 'rb'))
  tf = pickle.load(open('amazon_task/artifacts/tf.pkl', 'rb'))
  dt = pickle.load(open('amazon_task/artifacts/dt.pkl', 'rb'))
  svc = pickle.load(open('amazon_task/artifacts/svc.pkl', 'rb'))
except ModuleNotFoundError as e:
  st.error(f"Error loading models: {e}")
  st.stop()

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
  

st.title('Product Feedback Prediction')  

user_input = st.text_area("Enter your feedback:", "I don't like this product.")

if st.button('Predict'):
  prediction = predict_feedback(user_input)
  # No need to write prediction here as it's just the function call

st.markdown(
    """
    <style>
    .main { 
        background-color: #f0f2f6; 
    }
    h1 {
        color: #800080; /* Changed to Purple */
    }
    .stButton>button {
        background-color: #800080; /* Changed to Purple */
        color: white;
    }
    .stTextInput>div>input {
        background-color: #ffffff; 
        color: #800080; /* Changed to Purple */
        border: 2px solid #800080; /* Changed to Purple */
        padding: 10px;
    }
    .stTextInput>div>input:focus {
        border-color: #800080; /* Changed to Purple */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div style='text-align: center; color: #800080; font-size: 12px;'>
        <p>Created by Ahmed ramy</p>
    </div>
    """,
    unsafe_allow_html=True,
)
