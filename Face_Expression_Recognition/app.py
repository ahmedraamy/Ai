import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = load_model('Face_Expression_Recognition/artifacts/model.keras')

# List of class names (you can modify this list based on your dataset)
class_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit UI
st.markdown(
    "<h1 style='color: red;'>Facial Expression Recognition</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color: white; font-size: 18px;'>Upload an image to predict the facial expression</p>",
    unsafe_allow_html=True
)


# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image for prediction
    image = Image.open(uploaded_file)
    img = preprocess_image(image)

    # Make the prediction
    pred = model.predict(img)
    predicted_class = np.argmax(pred, axis=1)

    # Display the result (prediction above the image)
    st.write(f"Predicted Class: {class_names[predicted_class[0]]}")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image.", use_container_width=True)
    


st.markdown(
    """
    <div style='text-align: center; color: #800080; font-size: 12px;'>
        <p>Created by Ahmed ramy</p>
    </div>
""",
    unsafe_allow_html=True,
)
