from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import plotly.express as px
import base64
import os
import io
from PIL import Image

def app():
    model = load_model('Diabetes')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def predict(model, input_df):
        predictions_df = predict_model(estimator=model, data=input_df)
        predictions = predictions_df['prediction_label'][0]
        return predictions

   
    st.title("Early Diabetes Derecion App")
    st.sidebar.markdown("""# Select features

""")
    
   
    Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=80, value=0)
    Glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=80, value=0)
    BloodPressure = st.sidebar.number_input('BloodPressure', min_value=0, max_value=80, value=0)
    SkinThickness = st.sidebar.number_input('SkinThickness', min_value=0, max_value=80, value=0)
    Insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=80, value=0)
    BMI = st.sidebar.number_input('BMI', min_value=0, max_value=80, value=0)

    DiabetesPedigreeFunction = st.sidebar.number_input('DiabetesPedigreeFunction', min_value=0, max_value=80, value=0)
    Age = st.sidebar.number_input('Age', min_value=0, max_value=80, value=0)
    

    output=""
    action=""

    input_dict = {'Pregnancies' : Pregnancies, 'Glucose' : Glucose, 'BloodPressure' : BloodPressure, 'SkinThickness' : SkinThickness, 'Insulin' : Insulin, 'BMI' : BMI
    ,'DiabetesPedigreeFunction' : DiabetesPedigreeFunction, 'Age' : Age}
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
            
        if output==1:
            output = 'Arlet ! This patient has a high chances of being diabetic.' + '  ' +'The predicted outcome is :'+' '+str(output)
            action="Visit the doctor as soon as possible"
            st.error(output + '  ' + action)
        else:
            output = 'Great ! This patient has low chances of being diabetic.' + '  ' +'The predicted outcome is :'+' '+str(output)
            st.success(output + '  ' + action)

    else:
        st.write("")
        st.info('Awaiting for The prediction.')
        st.image('11.png', width=700)