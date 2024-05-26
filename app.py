import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("/content/drive/MyDrive/CODESOFT Data Sets/Titanic-Dataset.csv")

# Preprocess data
labelencoder = LabelEncoder()
df['Sex'] = labelencoder.fit_transform(df['Sex'])
df = df.drop(columns=['Cabin', 'Embarked'])
X = df[['Pclass', 'Sex']]
Y = df['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Train model
log = LogisticRegression(random_state=0)
log.fit(X_train, Y_train)

# Model accuracy
acc = accuracy_score(Y_test, log.predict(X_test))

# Streamlit interface
st.title("Titanic Survival Prediction")
st.write(f"Model Accuracy: {acc:.2f}")

st.sidebar.header("Input Features")
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=1)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

# Encode sex input
sex_encoded = 1 if sex == "Female" else 0

if st.sidebar.button("Predict"):
    res = log.predict([[pclass, sex_encoded]])
    if res == 0:
        st.write("The probability of survival is low.")
    else:
        st.write("The probability of survival is high.")

# Display data and model
if st.checkbox("Show Raw Data"):
    st.write(df)
