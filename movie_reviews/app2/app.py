import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Initialize models to None
tf_model, lr, dt, svc = None, None, None, None

# Load models
try:
    tf_model = pickle.load(open('artifacts/tf.pkl', 'rb'))  # TensorFlow model
    lr = pickle.load(open('artifacts/lr.pkl', 'rb'))        # Logistic Regression model
    dt = pickle.load(open('artifacts/dt.pkl', 'rb'))        # Decision Tree model
    svc = pickle.load(open('artifacts/svc.pkl', 'rb'))      # Support Vector Classifier
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")

# Ensure at least one model is loaded
if not (tf_model or lr or dt or svc):
    st.error("No models could be loaded. Please check the model files and paths.")
    st.stop() 
# Stopwords setup
stop_words = stopwords.words('english')
stop_words.remove('not')
stop_words.remove('no')

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Text preprocessing function
def text_preprocessing(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

# Predict feedback
def predict_feedback(text):
    processed_text = text_preprocessing(text)
    sentiment_score = analyzer.polarity_scores(processed_text)

    sentiment = 'Positive' if sentiment_score['compound'] > 0 else 'Negative'
    st.markdown(
        f"<p style='color: red; font-weight: bold;'>The review is <b>{sentiment}</b></p>",
        unsafe_allow_html=True,
    )
    return sentiment

# Streamlit App
st.title('Movie Reviews App')

user_input = st.text_area("Enter the text for movie review:", "I don't like this product.")

if st.button('Predict Sentiment'):
    predict_feedback(user_input)

# Add styles
st.markdown(
    """
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #1E90FF; }
    .stButton>button { background-color: #1E90FF; color: white; }
    .stTextInput>div>input {
        background-color: #ffffff; color: #1E90FF; border: 2px solid #1E90FF; padding: 10px;
    }
    .stTextInput>div>input:focus { border-color: #1E90FF; }
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
