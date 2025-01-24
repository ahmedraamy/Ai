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


# Download necessary NLTK resources
nltk.download('punkt')  # Corrected from 'punkt_tab' to 'punkt'
nltk.download('stopwords')
nltk.download('wordnet')  # Add this line to download WordNet
nltk.download('omw-1.4')  # Add this line to download Open Multilingual Wordnet
# Load model
model = load_model('movie_reviews/app1/artifacts/model.keras')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

with open('movie_reviews/app1/artifacts/tokenizer.pkl', 'wb') as f:
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
correct the code Traceback:
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 88, in exec_func_with_error_handling
    result = func()
             ^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 579, in code_to_exec
    exec(code, module.__dict__)
File "/mount/src/ai/movie_reviews/app1/app.py", line 68, in <module>
    prediction = predict_feedback(user_input)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/ai/movie_reviews/app1/app.py", line 46, in predict_feedback
    processed_text = text_preprocessing(text)
                     ^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/ai/movie_reviews/app1/app.py", line 39, in text_preprocessing
    text = word_tokenize(text)
           ^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/nltk/tokenize/__init__.py", line 142, in word_tokenize
    sentences = [text] if preserve_line else sent_tokenize(text, language)
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/nltk/tokenize/__init__.py", line 119, in sent_tokenize
    tokenizer = _get_punkt_tokenizer(language)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/nltk/tokenize/__init__.py", line 105, in _get_punkt_tokenizer
    return PunktTokenizer(language)
           ^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/nltk/tokenize/punkt.py", line 1744, in __init__
    self.load_lang(lang)
File "/home/adminuser/venv/lib/python3.12/site-packages/nltk/tokenize/punkt.py", line 1749, in load_lang
    lang_dir = find(f"tokenizers/punkt_tab/{lang}/")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/nltk/data.py", line 579, in find
    raise LookupError(resource_not_found)
