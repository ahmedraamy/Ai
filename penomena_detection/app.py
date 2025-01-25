import streamlit as st
<<<<<<< HEAD
import numpy as np
import tensorflow as tf
=======
import tensorflow as tf
import numpy as np
>>>>>>> 453a19f3fe09a2db95ce040d3cc808cc2e54c915
from PIL import Image, ImageOps

# Class dictionary
dic = {0: 'Normal', 1: "Infected"}

<<<<<<< HEAD
# Load the pre-trained Keras model (for comparison)
# Uncomment if using the Keras model
# gdown.download(model_url, model_output, quiet=False)
# model = tf.keras.models.load_model(model_output)

# Load the pre-trained TFLite model
interpreter = tf.lite.Interpreter(model_path="artifacts/converted_model.tflite")
interpreter.allocate_tensors()

# Helper function to preprocess image
def preprocess_image(img):
    img = Image.open(img)
    # Check if the image is grayscale, if not, convert
    if len(np.array(img).shape) != 2:
        img = ImageOps.grayscale(img)
    img = img.resize((224, 224))  # Resize image to 224x224
    img_array = np.expand_dims(np.array(img), axis=-1)  # Add channel dimension
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Display Title and Upload Image
=======
# Load the pre-trained TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('artifacts/converted_model.tflite', 'wb') as f:
    f.write(tflite_model)
# Upload image
>>>>>>> 453a19f3fe09a2db95ce040d3cc808cc2e54c915
st.markdown(
    "<h1 style='color: #d20000;'>Pneumonia Detection from X-ray</h1>",
    unsafe_allow_html=True
)
img_file = st.file_uploader("Upload your X-ray", type=["jpg", "jpeg", "png"])

<<<<<<< HEAD
# Display uploaded image
if img_file:
    st.image(img_file, caption="Uploaded Image", use_column_width=True)

# Prediction
if img_file and st.button("Predict"):
    try:
        # Preprocess image
        img_array = preprocess_image(img_file)
        
        # If using the Keras model:
        # pred = model.predict(img_array)
        # result = "Pneumonia" if pred[0] > 0.5 else "Healthy"

        # Using the TFLite model for prediction
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = output_data[0][0]
        result = dic[int(pred)]  # Mapping the prediction to the class

        # Display result
        st.success(f"The prediction is: {result}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Branding and Footer
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 14px; color: gray;">
        Created by <strong>Ahmed Ramy</strong> 
    </div>
    """, unsafe_allow_html=True
=======
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
        interpreter.invoke()

        # Get the output tensor and make the prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = output_data[0][0]  # Assuming the model returns a single probability value

        # Map the prediction to the class label
        result = dic[int(pred)]  # Convert the prediction to the class label

        # Display the result
        st.success(f"The prediction is: {result}")

    except Exception as e:
        st.error(f"An error occurred: {e}")  # Ensures exception is caught and displayed

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #d20000; font-size: 12px;'>
        <p>Created by Ahmed Ramy</p>
    </div>
""",
    unsafe_allow_html=True,
>>>>>>> 453a19f3fe09a2db95ce040d3cc808cc2e54c915
)
