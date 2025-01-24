import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Ensure necessary NLTK packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load stop words and lemmatizer
stop_words = stopwords.words('english')
stop_words.remove('not')
stop_words.remove('no')

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Directory for your model files
model_directory = 'movie_reviews/app2/artifacts/'

# Load pre-trained models (if available)
try:
    # Check if the required model files exist before loading
    if not os.path.exists(os.path.join(model_directory, 'lr.pkl')):
        st.error("Logistic Regression model (lr.pkl) not found!")
        st.stop()

    if not os.path.exists(os.path.join(model_directory, 'tf.pkl')):
        st.error("TF-IDF model (tf.pkl) not found!")
        st.stop()

    if not os.path.exists(os.path.join(model_directory, 'dt.pkl')):
        st.error("Decision Tree model (dt.pkl) not found!")
        st.stop()

    if not os.path.exists(os.path.join(model_directory, 'svc.pkl')):
        st.error("SVM model (svc.pkl) not found!")
        st.stop()

    lr = pickle.load(open(os.path.join(model_directory, 'lr.pkl'), 'rb'))
    tf = pickle.load(open(os.path.join(model_directory, 'tf.pkl'), 'rb'))
    dt = pickle.load(open(os.path.join(model_directory, 'dt.pkl'), 'rb'))
    svc = pickle.load(open(os.path.join(model_directory, 'svc.pkl'), 'rb'))

except FileNotFoundError as e:
    st.error(f"Error loading model: {e}. Make sure the model files are in the correct directory.")
    st.stop()

# Text preprocessing function
def text_preprocessing(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = word_tokenize(text)  # Tokenize text into words
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize words
    text = ' '.join(text)  # Join words back into a string
    return text

# Sentiment prediction function
def predict_feedback(text):
    processed_text = text_preprocessing(text)  # Preprocess the text
    sentiment_score = analyzer.polarity_scores(processed_text)  # Analyze sentiment
    
    # Determine sentiment based on compound score
    if sentiment_score['compound'] > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    # Display the sentiment result on Streamlit
    st.markdown(
        f"<p style='color: red; font-weight: bold;'>The review is <b>{sentiment}</b></p>",
        unsafe_allow_html=True,
    )

# Streamlit UI
st.title('Movie Reviews App')

# User input text area
user_input = st.text_area("Enter the text for movie review:", "I don't like this product.")

# Button to trigger sentiment prediction
if st.button('Predict Sentiment'):
    prediction = predict_feedback(user_input)

# Custom CSS for styling
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
