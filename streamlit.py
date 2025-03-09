import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Tạo dữ liệu giả định để huấn luyện mô hình
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Dữ liệu đầu vào ngẫu nhiên
y = 2.5 * X + np.random.randn(100, 1) * 2  # Giá trị mục tiêu với một chút nhiễu

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên bộ dữ liệu test
y_pred = model.predict(X_test)

# Tính toán lỗi
mse = mean_squared_error(y_test, y_pred)

# Tạo ứng dụng Streamlit
st.title('Ứng dụng Dự đoán với Linear Regression')

st.write(f'Độ lỗi bình quân (MSE): {mse:.2f}')

# Cho phép người dùng nhập dữ liệu
user_input = st.number_input("Nhập giá trị X để dự đoán Y:", min_value=0.0, max_value=10.0, value=5.0)

# Dự đoán dựa trên giá trị người dùng nhập
prediction = model.predict(np.array([[user_input]]))
st.write(f'Dự đoán giá trị Y tương ứng với X = {user_input} là: {prediction[0][0]:.2f}')
