
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import gdown

# Direct link to the Google Drive file
url = 'https://drive.google.com/uc?id=1NaEyY4hKFUN6znsWIZoh3LrizB5xr990'
output = 'Model.keras'

# Download the model from Google Drive using gdown
gdown.download(url, output, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(output)

# Upload an X-ray image from the user
img = st.file_uploader('Upload your X-ray', type=['jpg', 'png', 'jpeg'])

# Check if an image has been uploaded
if img is not None:
    st.image(img)  # Display the uploaded image

    # Button to trigger prediction
    button = st.button('Predict')

    if button:
        # Open the uploaded image and convert it to grayscale (single channel)
        img = Image.open(img)
        img = img.convert("L")  # Convert to grayscale (1 channel)
        
        # Resize the image to the input size expected by the model
        img = img.resize((256, 256))
        
        # Normalize the image to [0, 1] range and expand dimensions for batch size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 256, 256, 1)
        
        # Predict the class (Pneumonia or Normal) using the model
        pred = model.predict(img_array)
        
        # If the prediction value is greater than 0.5, consider the person as likely having Pneumonia
        if pred[0] > 0.5:
            st.write("The person is likely to have Pneumonia.")
        else:
            st.write("The person is likely to be healthy.")
else:
    st.write("Please upload an image.")


st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 14px; color: gray;">
        This page was created by <strong>Mohamed Hosam</strong>
    </div>
    """, unsafe_allow_html=True)
