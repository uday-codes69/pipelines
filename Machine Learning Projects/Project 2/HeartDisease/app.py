import streamlit as st
import pandas as pd
import joblib
import joblib

model = joblib.load("/Users/udaythakur/Machine Learning Projects/Project 2/HeartDisease/KNN_heart.pkl")
scaler = joblib.load('/Users/udaythakur/Machine Learning Projects/Project 2/HeartDisease/scaler.pkl')
expected_columns = joblib.load('/Users/udaythakur/Machine Learning Projects/Project 2/HeartDisease/columns.pkl')

st.title("Heart Stroke Prediction❤️")
st.markdown("Provide the following Details")

age  = st.slider('Age',18,100)
sex = st.selectbox("Sex", ["M", "F"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chest_pain = st.selectbox("Chest Pain Type", ["ATA","NAP","TA","ASY"])
cholesterol = st.number_input("CHolesterol (mg/dL)",100,600,200)
resting_ecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
max_hr = st.slider("Max Heart Rate",60,220,150)
exercise_angina = st.selectbox("Exercise-Induced Angina",["Y","N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button("Predict"):
    raw_input = {
        'Age' : age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    
    #create input dataframes
    input_df = pd.DataFrame([raw_input])
    
    
    #Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
        
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")