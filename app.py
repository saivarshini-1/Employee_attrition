import streamlit as st
import joblib
import pandas as pd
import numpy as np


# Load the model and label encoder
model = joblib.load("attrition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Employee Attrition prediction")

st.sidebar.header("Employee Details")

def get_user_input():
    inputs = {}
    inputs['Age'] = st.sidebar.number_input("Age", min_value=18, max_value=65, value = 30)
    inputs['MonthlyIncome']= st.sidebar.number_input(
        "Monthly Income",min_value = 1000, max_value = 20000, value=5000
    )
    inputs['JobSatisfaction'] =st.sidebar.selectbox(
        "Job Satisfaction" ,options=[1,2,3,4]
    )
    inputs['OverTime'] =st.sidebar.selectbox(
        "Over Time" ,options=["Yes","No"]
    )
    inputs['DistanceFromHome'] =st.sidebar.number_input(
        "Distance From Home" , min_value=0, max_value=50, value=10
    )

    data ={}
    for feat in feature_columns:
       if feat in inputs:
           data[feat] = inputs[feat]
       else:
          data[feat] = 0
    return pd.DataFrame(data,index=[0])

user_input = get_user_input()
user_input['OverTime'] = label_encoder.transform(
    user_input['OverTime'])

#predict the attrition
if st.button("Predict Attrition"):
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[0][1]

    if prediction[0] ==1:
        st.error("The employee is likely to leave the company.")

    else:
        st.success("The employee is likely to stay with company.")
    st.info(f"Prediction Probability: {probability:.2f}")



    
    
    

