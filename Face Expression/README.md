# Facial Expression Recognition Project 😃

This project is focused on **Facial Expression Recognition** using deep learning. The model is built using **Convolutional Neural Networks (CNN)** to classify facial expressions into 7 categories. The dataset used is the **FER-2013 dataset**, which is available on Kaggle. The project aims to detect human facial emotions for applications in areas such as human-computer interaction, healthcare, and security.

## Features 🌟

- **Facial Expression Classification**: The model can predict one of the following emotions:
  - 😡 Anger
  - 🤢 Disgust
  - 😱 Fear
  - 😀 Happy
  - 😞 Sad
  - 😲 Surprise
  - 😐 Neutral
  
- **Preprocessing**: The images are preprocessed with:
  - Image resizing 🖼️
  - Normalization for scaling pixel values between 0 and 1.
  - Data augmentation to improve model generalization.

- **Model**: The model is a **deep learning CNN** architecture with multiple convolutional layers to extract important features from facial expressions.

## Technologies Used 🛠️

- **TensorFlow/Keras**: For building and training the deep learning model.
- **NumPy**: For numerical computing and handling image arrays.
- **Matplotlib**: For data visualization and plotting training curves.
- **Streamlit**: For deploying the model as a web application.

## Project Structure 📂

facial-expression-recognition/
├── app.py                # Main script to run the Streamlit app
├── model/                # Folder containing the trained model (best.keras)
├── requirements.txt      # List of dependencies for the project
├── README.md             # Project documentation (this file)


## Installation Instructions 📥

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/AhmedKamel200058/Face_Recognition.git
   ```
2. Create a virtual environment (optional but recommended):
	```bash
	conda create --name tensorflow_env python=3.11
	```


3. Activate the virtual environment:
	```bash
	conda activate tensorflow_env
	```
	
4. Install the required dependencies:
	```bash
	 pip install -r requirements.txt
	```
## How to Run the Application 🚀
After setting up the environment, run the Streamlit app:
```bash
streamlit run app.py
```

## Usage 📸

1. Upload a facial image in .jpg, .jpeg, or .png format.
2. The model will predict the facial expression and display the result.

## Future Improvements 🚀
1. Improve model accuracy with more training data.
2. Add real-time webcam facial expression recognition.
3. Integrate with other systems for automatic emotion detection in videos.


## Acknowledgements 🙏
FER-2013 Dataset: Available on Kaggle for training the model.
TensorFlow and Keras for providing the deep learning frameworks.
Streamlit for creating easy-to-deploy web applications.
