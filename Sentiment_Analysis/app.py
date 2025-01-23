import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

# Function to load models and vectorizers
def load_model():
    # Load the model and vectorizer
    model = pickle.load(
        open("artifacts/model.pkl", "rb")
    )  # Use forward slashes for paths
    vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))
    return model, vectorizer

# Load model and vectorizer
model, vectorizer = load_model()

# Function to predict sentiment
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Set up Streamlit interface
st.set_page_config(
    page_title="Sentiment Analysis", page_icon=":chart_with_upwards_trend:"
)

# Add custom styling for better appearance
st.markdown(
    """
    <style>
    .main { 
        background-color: #f0f2f6;  /* Keep this if you want a light background */
    }
    h1 {
        color: #1E90FF;  /* Change heading color to blue */
    }
    .stButton>button {
        background-color: #1E90FF;  /* Blue button */
        color: white;
    }
    .stTextInput>div>input {
        background-color: #ffffff;  /* White background for input */
        color: #1E90FF;  /* Blue text in the input */
        border: 2px solid #1E90FF;  /* Blue border */
        padding: 10px;
    }
    .stTextInput>div>input:focus {
        border-color: #1E90FF;  /* Blue border when focused */
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Display app header
st.title("Sentiment Analysis App")


# Add input field for user text
text_input = st.text_area("Enter the text for sentiment analysis:" , "I don't like it.")

# Prediction button
if st.button("Analyze Sentiment"):
    if text_input:
        sentiment = predict_sentiment(text_input)
        if sentiment == 1:
            # Show Positive sentiment in red color
            st.markdown(
                "<p style='color: red; font-weight: bold;'>The sentiment is <b>Positive</b> ",
                unsafe_allow_html=True,
            )
        else:
            # Show Negative sentiment in white color
            st.markdown(
                "<p style='color: red; font-weight: bold;'>The sentiment is <b>Negative</b> ",
                unsafe_allow_html=True,
            )
    else:
           st.markdown(
                "<p style='color: red; font-weight: bold;'>Please enter some text to <b>analyze</b> ",
                unsafe_allow_html=True,
            )
       # st.warning("Please enter some text to analyze.")

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #1E90FF; font-size: 12px;'>
        <p>Created by Ahmed ramy</p>
    </div>
""",
    unsafe_allow_html=True,
)
