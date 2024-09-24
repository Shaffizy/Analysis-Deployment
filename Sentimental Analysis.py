# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:33:24 2024

@author: Hp
"""

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st


loaded_model = pickle.load(open('C:/Users/Hp/Downloads/trained_model.sav', 'rb'))

# Load the tokenizer used during training
with open('C:/Users/Hp/Downloads/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment(review):
    # Tokenize and pad the review using the trained tokenizer
    sequence = tokenizer.texts_to_sequences([review])
  
    padded_sequence = pad_sequences(sequence, maxlen=200)
    
    # Make a prediction
    prediction = loaded_model.predict(padded_sequence)
    
    # Determine sentiment
    sentiment = "positive" if prediction[0][0] > 0.6 else "negative"
    return sentiment

def main():
    # giving a title
    st.title('Sentimental Analysis Web App')
    
     # getting the input data from the user
    Reviews = st.text_input('Make a Review.')
    
    # code for Prediction
    sentiment = ''
    
    # creating a button for Prediction
    if st.button('Review Sentiment'):
        sentiment = predict_sentiment(Reviews)
            
           
    st.success(sentiment) 

if __name__ == '__main__':
    main()