import helper
import streamlit as st
import pickle


model = pickle.load(open("amazon_task/artifacts/lr.pkl", 'rb'))



text = st.text_input('enter your review')

text = helper.text_preprocessing(text)
pred = model.predict(text)

if st.button("predict") :
    st.text(pred)
