# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 20:34:04 2023

@author: zaidi
"""

import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('D:/project for deployment/trained_model.sav', 'rb') as file:
        model = pickle.load(file)
    return model

def predict_price(model, features):
    prediction = model.predict(features)
    return prediction

def main():
    st.title("Electricity Price Prediction")

    # Load the model
    model = load_model()

    # User input for features
    feature1 = st.number_input("Enter Day:")
    feature2 = st.number_input("Enter Month:")
    feature3 = st.number_input("Enter ForecastWindProduction:")
    feature4 = st.number_input("Enter SystemLoadEA:")
    feature5 = st.number_input("Enter SMPEA:")
    feature6 = st.number_input("Enter ORKTemperature:")
    feature7 = st.number_input("Enter ORKWindspeed:")
    feature8 = st.number_input("Enter CO2Intensity:")
    feature9 = st.number_input("Enter ActualWindProduction:")
    feature10 = st.number_input("Enter SystemLoadEP2:")

    # Prepare input features for prediction
    features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10]])

    # Make prediction using the model
    prediction = predict_price(model, features)

    # Display prediction
    st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()

