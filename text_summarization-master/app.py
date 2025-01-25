import streamlit as st
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import pickle
from attention import AttentionLayer


max_text_len=30
max_summary_len=8

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


model = load_model('model.h5', custom_objects={'AttentionLayer': AttentionLayer})

json_file = open('encoder_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder_model = model_from_json(loaded_model_json)

json_file = open('decoder_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder_model = model_from_json(loaded_model_json, custom_objects={'AttentionLayer': AttentionLayer})

with open('x_tokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)
with open('y_tokenizer.pickle', 'rb') as handle:
    y_tokenizer = pickle.load(handle)

reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

# Decode sequence
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

# summarize function
def summarize(text):
    text = pad_sequences(x_tokenizer.texts_to_sequences([text]),maxlen=max_text_len, padding='post')
    return decode_sequence(text.reshape(1,max_text_len))



def main():
	st.title("Text Summarization using Deep Learning")
	st.text("Summarizing your text")
	message = st.text_area("Enter your text","please type here")
	if st.button("Summarize"):
		summary_result = summarize(message)
		st.success(summary_result) 
	st.sidebar.subheader("Text Summarizer")
	st.sidebar.text("NLP app deployed with streamlit")

if __name__ == '__main__':
	main()