import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
try:
    with open(r"C:\Users\user\Desktop\projects\projct/lr.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'lr.pkl' exists in the 'projct' directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Streamlit app title and description
st.title("Diabetes Prediction App")
st.markdown("""
This app predicts whether a patient is likely to have diabetes based on their input parameters. 
Please fill in the details below.
""")

# Input fields
st.header("Enter Patient Details:")
pregnancies = st.number_input("Number of Pregnancies:", min_value=0, step=1, help="Number of times the patient has been pregnant.")
glucose = st.number_input("Glucose Level:", min_value=0, help="Blood glucose concentration (mg/dL).")
insulin = st.number_input("Insulin Level (IU/ml):", min_value=0, help="Insulin concentration (μU/ml).")
bmi = st.number_input("Body Mass Index (BMI):", min_value=0.0, format="%.2f", help="BMI is a measure of body fat based on height and weight.")
dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0, format="%.3f", help="A function which scores likelihood of diabetes based on family history.")
age = st.number_input("Age:", min_value=0, step=1, help="Age of the patient in years.")

# Prediction logic
if st.button("Predict"):
    # Ensure that all inputs are reasonable
    if glucose == 0 or bmi == 0:
        st.warning("Glucose and BMI values must be greater than zero for a meaningful prediction.")
    else:
        try:
            # Prepare input data for prediction
            input_data = np.array([[pregnancies, glucose, insulin, bmi, dpf, age]])
            
            # Predict using the loaded model
            prediction = model.predict(input_data)
            
            # Display the result
            if prediction[0] == 1:
                st.error("The patient is likely to have diabetes.")
            else:
                st.success("The patient is not likely to have diabetes.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
