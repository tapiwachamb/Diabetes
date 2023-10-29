
# Early Diabetes Prediction

##  *A machine learning model  for Early Diabetes detection*


![APP](https://drive.google.com/uc?id=1qd7Og6ij_KCO_0n8s7n8Y3CxSURyPqFn&export=download)


## TAPIWA CHAMBOKO
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://tapiwachamb.github.io/tapiwachamboko.io/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tapiwa-chamboko-327270208/)
[![github](https://img.shields.io/badge/github-1DA1F2?style=for-the-badge&logo=githubr&logoColor=white)](https://github.com/tapiwachamb)


## 🚀 About Me
I'm a full stack developer experienced in deploying artificial intelligence powered apps


## Authors

- [@Tapiwa chamboko](https://github.com/tapiwachamb)


## Demo

**Live demo**

[Click here for Live demo](https://diabetes-app.streamlit.app/)
## Installation

Install required packages 

```bash
  pip install streamlit
  pip install pycaret
  pip insatll scikit-learn==0.23.2
  pip install numpy
  pip install seaborn 
  pip install pandas
  pip install matplotlib
  pip install plotly-express
  pip install streamlit-lottie
```
    
## Datasets
- [Download Customer Churn Datasets here](https://www.kaggle.com/mathchi/diabetes-data-set)
## Data
- Train_data folder contains the data for training the model
- Test_data folder conain tha the data for testing the model 


## Model Notebook
- *Model notebook is in note_book folder*
- run the notebook in jupyter notebook or [google colab](https://colab.research.google.com/)


**Example code**
- import libraries

```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
%matplotlib inline
import plotly.express as px
```
- load data in dataframe
```bash
df = pd.read_csv("Train_data/diabetes.csv",header = 0)
df.head()
```
- data preparation
```bash
from pycaret.classification import *
s = setup(df, target = 'Outcome', session_id = 8789, use_gpu = True)
```
- Random Forest Classifier
```bash
rf = create_model('rf', bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=8789, verbose=True,
                       warm_start=False)
```
## Deployment

To deploy this project we used streamlit to create Web App
- Run this code below

```bash
  streamlit run app.py 
```


## Model Pipeline

```bash
ransformation Pipeline and Model Successfully Saved
(Pipeline(memory=None,
          steps=[('dtypes',
                  DataTypes_Auto_infer(categorical_features=[],
                                       display_types=True, features_todrop=[],
                                       id_columns=[],
                                       ml_usecase='classification',
                                       numerical_features=[], target='Outcome',
                                       time_features=[])),
                 ('imputer',
                  Simple_Imputer(categorical_strategy='not_available',
                                 fill_value_categorical=None,
                                 fill_value_numerical=None,
                                 numeric_stra...
                  RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                         class_weight=None, criterion='gini',
                                         max_depth=None, max_features='auto',
                                         max_leaf_nodes=None, max_samples=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1, min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         n_estimators=100, n_jobs=-1,
                                         oob_score=False, random_state=8789,
                                         verbose=0, warm_start=False)]],
          verbose=False),
 'Diabetes.pkl')

```


## Deployed Model
- The Deployed model pipeline  is named Diabetes.pkl
- This pipeline wil be used in the web app
## Appendix

Happy Coding!!!!!!


