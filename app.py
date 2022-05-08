import streamlit as st
import pandas as pd
import numpy as np
import keras
import joblib
from sklearn.ensemble import RandomForestRegressor
from prediction import get_prediction
#from keras.models import load_model
from load_model import get_model

rf_model = get_model(model_path = r'Model/CyberBully_DL_Model.h5')

st.set_page_config(page_title="CyberBully Sentiment Prediction",
                   page_icon="🚧", layout="wide")


st.markdown("<h1 style='text-align: center;'>CyberBully Sentiment Prediction 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter a tweet text to predict sentiment:")
        text = st.text_input("text", value="", max_chars=100)
        
        submit = st.form_submit_button("Predict")


    if submit:
        data = text

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted sentiment for given text is:  {pred}")

if __name__ == '__main__':
    main()
