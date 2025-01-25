import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Class dictionary
dic = {0: 'Normal', 1: "Infected"}

# Load the pre-trained TFLite model
interpreter = tf.lite.Interpreter(model_path="artifacts/converted_model.tflite")
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

        # Check dimensions and convert to grayscale if necessary
        if len(np.array(img).shape) != 2:
            st.warning("Image is not grayscale. Converting to grayscale.")
            img = ImageOps.grayscale(img)

        # Resize to 224x224
        img = img.resize((224, 224))
        
        # Convert to NumPy array and add channel dimension (for grayscale)
        img_array = np.expand_dims(np.array(img), axis=-1)

        # Normalize the image and add batch dimension
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

        # Invoke the interpreter
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = output_data[0][0]

        # Map prediction to class label
        result = dic[int(pred)]

        # Display result
        st.success(f"The prediction is: {result}")

    except Exception as e:
        st.error(f"An error occurred: {e}")


st.markdown(
    """
    <div style='text-align: center; color: #d20000; font-size: 12px;'>
        <p>Created by Ahmed Ramy</p>
    </div>
""",
    unsafe_allow_html=True,
)
