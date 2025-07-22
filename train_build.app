import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("adult.csv")

if 'educational-num' in data.columns:
    data.rename(columns={'educational-num': 'education-num'}, inplace=True)

data.workclass.replace("?", "Not Listed", inplace=True)
data.occupation.replace("?", "Others", inplace=True)

if 'education' in data.columns:
    data.drop(columns=["education"], inplace=True)

cat_cols = ['workclass','marital-status','occupation','relationship','race','gender','native-country']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop(columns=['income'])
y = data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, pipeline.predict(X_test)))

joblib.dump(pipeline, "best_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "feature_order.pkl")

%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("💰 Employee Salary Prediction")

st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, value=200000)
education_num = st.sidebar.slider("Education Number", 1, 16, 9)
marital = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
race = st.sidebar.selectbox("Race", encoders['race'].classes_)
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.sidebar.number_input("Capital Gain",min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss",min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
country = st.sidebar.selectbox("Native Country", encoders['native-country'].classes_)

input_dict = {
    'age': age,
    'workclass': encoders['workclass'].transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'education-num': education_num,
    'marital-status': encoders['marital-status'].transform([marital])[0],
    'occupation': encoders['occupation'].transform([occupation])[0],
    'relationship': encoders['relationship'].transform([relationship])[0],
    'race': encoders['race'].transform([race])[0],
    'gender': encoders['gender'].transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encoders['native-country'].transform([country])[0],
}

df = pd.DataFrame([input_dict])

st.write("### 👇 Your Input Data")
st.dataframe(df)

if st.button("Predict Salary Class"):
    prediction = model.predict(df)[0]
    st.markdown(f"### 🎯 **Prediction: `{prediction} $`**")
from pyngrok import ngrok
import os
import time

os.system("streamlit run app.py &")
time.sleep(5)
public_url = ngrok.connect(8501)
print("Streamlit app live at:", public_url)
