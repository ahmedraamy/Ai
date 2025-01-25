import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Class dictionary
dic = {0: 'Normal', 1: "Infected"}

# Load the pre-trained TFLite model
interpreter = tf.lite.Interpreter(model_path="penomena_detection/artifacts/converted_model.tflite")
interpreter.allocate_tensors()

# Upload image
st.markdown(
    "<h1 style='color: #d20000;'>Pneumonia Detection from X-ray</h1>",
    unsafe_allow_html=True
)
img_file = st.file_uploader("Upload your X-ray", type=["jpg", "jpeg", "png"])

# Display image
if img_file:
    st.image(img_file, caption="Uploaded Image", use_column_width=True)

# Predict button
if img_file and st.button("Predict"):
    try:
        # Open the image
        img = Image.open(img_file)

        # Convert to grayscale if the image is not already
        if len(np.array(img).shape) != 2:
            st.warning("Image is not grayscale. Converting to grayscale.")
            img = ImageOps.grayscale(img)

        # Resize the image to the size expected by the model (224x224)
        img = img.resize((224, 224))

        # Convert the image to a NumPy array and expand the dimensions (for grayscale)
        img_array = np.expand_dims(np.array(img), axis=-1)

        # Normalize the image data to [0, 1] range
        img_array = img_array / 255.0

        # Add batch dimension (required for TensorFlow Lite model)
        img_array = np.expand_dims(img_array, axis=0)

        # Get the input details from the model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor (ensure data type matches)
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

        # Run inference
      
