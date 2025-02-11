import streamlit as st
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

clf = joblib.load('diabetes_model.pkl')
df = pd.read_csv('cleaned_df.csv')

X = df.drop('diabetes',axis=1).to_numpy()
y = df['diabetes'].to_numpy()

st.write("""
# Diabetes Prediction App
This app predicts whether a person has **diabetes** based on input features!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    hypertension = st.sidebar.radio('Hypertension', ['No', 'Yes'])
    hypertension = 1 if hypertension == 'Yes' else 0
    heart_disease = st.sidebar.radio('Heart Disease', ['No', 'Yes'])
    heart_disease = 1 if heart_disease == 'Yes' else 0
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    HbA1c_level = st.sidebar.slider('HbA1c Level', 4.0, 15.0, 5.5)
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 50.0, 300.0, 120.0)

    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    gender_Female = 1 if gender == 'Female' else 0
    gender_Male = 1 if gender == 'Male' else 0

    age_group = st.sidebar.selectbox('Age Group', ['Child (age 0-18)', 'Young Adult (age 19-35)', 'Adult (age 36-60)', 'Senior (age 61+)'])
    age_group_Child = 1 if age_group == 'Child' else 0
    age_group_Young_Adult = 1 if age_group == 'Young Adult' else 0
    age_group_Adult = 1 if age_group == 'Adult' else 0
    age_group_Senior = 1 if age_group == 'Senior' else 0

    smoking_history_binary = st.sidebar.radio('Smoking History', ['No', 'Yes'])
    smoking_history_binary = 1 if smoking_history_binary == 'Yes' else 0

    data = {
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level,
        'gender_Female': gender_Female,
        'gender_Male': gender_Male,
        'age_group_Child': age_group_Child,
        'age_group_Young Adult': age_group_Young_Adult,
        'age_group_Adult': age_group_Adult,
        'age_group_Senior': age_group_Senior,
        'smoking_history_binary': smoking_history_binary
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features().to_numpy()

st.subheader('User Input Parameters')
st.write(df)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction')
st.write('Diabetes' if prediction[0] == 1 else 'No Diabetes')

st.subheader('Prediction Probability')
st.write(prediction_proba)