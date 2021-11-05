# pip install openpyxl
import pandas as pd
import streamlit as st
import zipfile
import base64
import os

from PIL import Image
from multiapp import MultiApp
from apps import home, diabetes1, diabetes2 # import your app modules here
img = Image.open("android-chrome-384x384.png")
app = MultiApp()

st.set_page_config(page_icon=img)
#st.image('unnamed2.png', width=500)

st.sidebar.markdown("""# Diabetes Prediction App

""")
with st.sidebar.header(''):
    

    app.add_app("Home", home.app)
    #app.add_app("Visuals & Reprts", data.app)
    #app.add_app("Real Time Customer Churn", cstomer1.app)
    #app.add_app("Batch Customer Churn", customer2.app)
    app.add_app("Individual Prediction", diabetes1.app)
    app.add_app("Batch Prediction", diabetes2.app)
    
   # app.add_app("Predictions", model.app)
    
    
    
    
st.markdown("""




""")
primaryColor = st.get_option("theme.primaryColor")
s = f"""
<style>
div.stButton > button:first-child {{ border: 10px solid {primaryColor};background-color: #6C63FF;color:white;font-size:16px;height:3em;width:18.5em; border-radius:5px 5px 5px 5px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)
# Add all your application here

# The main app
app.run()
