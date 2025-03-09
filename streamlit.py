import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

st.title("Scikit-learn and Streamlit Example")

# Load a dataset
dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

def get_dataset(name):
    data = None
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(pd.Series(y).unique()))

# Add model parameters
st.sidebar.header("Hyperparameters")
n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)

# Train the model
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Accuracy = {acc}")

# Display the dataset
st.subheader("Dataset Preview")
if dataset_name == "Iris":
    df = pd.DataFrame(data=X, columns=datasets.load_iris().feature_names)
    df['target'] = y
    st.dataframe(df)
elif dataset_name == "Breast Cancer":
    df = pd.DataFrame(data=X, columns=datasets.load_breast_cancer().feature_names)
    df['target'] = y
    st.dataframe(df)
else:
    df = pd.DataFrame(data=X, columns=datasets.load_wine().feature_names)
    df['target'] = y
    st.dataframe(df)
