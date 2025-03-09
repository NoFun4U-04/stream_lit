import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

st.title("Streamlit App with Scikit-learn")

# Example 1: Linear Regression with Diabetes Dataset
st.header("Linear Regression with Diabetes Dataset")

# Load the diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

st.write("Diabetes Dataset Preview:")
st.dataframe(df.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.write(f"Mean Squared Error: {mse}")

# Plot actual vs. predicted values
st.subheader("Actual vs. Predicted Values")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
st.pyplot(fig)

# Example 2: Simple Data Visualization (same as before)
st.header("Data Visualization")

# Generate some random data
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.rand(100),
    'y': np.random.rand(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Scatter plot
st.subheader("Scatter Plot")
fig, ax = plt.subplots()
sns.scatterplot(x='x', y='y', hue='category', data=data, ax=ax)
st.pyplot(fig)

# Histogram
st.subheader("Histogram")
fig, ax = plt.subplots()
sns.histplot(data['x'], ax=ax)
st.pyplot(fig)

# Example 3: User Input and Simple Calculations (same as before)
st.header("User Input and Calculations")

num1 = st.number_input("Enter a number:")
num2 = st.number_input("Enter another number:")

if st.button("Calculate Sum"):
    st.write(f"The sum is: {num1 + num2}")

# Example 4: Displaying Text and Markdown (same as before)
st.header("Text and Markdown")

st.write("This is a simple text display.")

st.markdown("""
    # Markdown Example
    You can use markdown to format text.
    * Bullet points
    * _Italics_
    * **Bold**
""")
