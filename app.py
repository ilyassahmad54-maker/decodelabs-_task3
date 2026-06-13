import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("diabetes.csv")

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# ---------------------------
# MODEL
# ---------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ---------------------------
# UI DASHBOARD
# ---------------------------
st.title("🏥 Diabetes Prediction Dashboard (Pakistan Health AI)")

st.sidebar.header("Enter Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose Level", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 100, 30)

input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict Diabetes Risk"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Patient may have Diabetes")
    else:
        st.success("✅ Low Risk: Patient is Healthy")

# ---------------------------
# MODEL PERFORMANCE
# ---------------------------
st.subheader("📊 Model Performance")
st.write("Accuracy:", acc)
st.write("Confusion Matrix:")
st.write(cm)

# ---------------------------
# DATA PREVIEW
# ---------------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())


st.markdown("🚀 Built for Decode Labs Internship | Ilyas Ahmad khan Lodhi")
