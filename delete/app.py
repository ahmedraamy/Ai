import streamlit as st
import helper
import pickle
from PIL import Image
import numpy as np


global prep_image

model = pickle.load(open("model59.pkl",'rb'))


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"]{
    background-size: auto; /* Prevents scaling with window size */
    transform: scale(1); /* Zoom level set to 180% */
    transform-origin: center; /* Keeps the zoom centered */
    height: 100vh;
    width: 100vw;
    margin: 0;
    padding: 0;
    overflow: hidden; }

    [data-testid="stHeader"]{

    }
    .center-content {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh; /* Full viewport height */
            flex-direction: column;
            position: relative;
            z-index: 1;
        }}
        .stTextInput, .stButton {{
            margin-top: 20px;
            width: 300px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# st.title("Sentiment Analysis Application using ML ✨") 

title = "<h1 style='text-align: center; color: #FF5733; white-space: nowrap;'>Emotion Detection using ML</h1>" 
st.markdown(title, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
st.markdown('<div class="center-content">', unsafe_allow_html=True)

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    prep_image = helper.preprocessing(image)
    # print(type(prep_image))
    prediction = np.argmax(model.predict(prep_image))

    # Display the image
    st.image(image, caption="Uploaded Photo")



state = st.button("Classify","review")


Class_Names= ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Create a centered button
if state:

    st.markdown(f"<h3 style='color: black; font-size:30px;font-family: 'Times New Roman', Times, serif;'>Emotion: {Class_Names[prediction]}</h3>", unsafe_allow_html=True)


