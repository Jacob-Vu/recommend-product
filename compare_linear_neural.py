# Import các thư viện cần thiết
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (chuẩn hóa cho mô hình mạng nơ-ron)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

# Khởi tạo mô hình Linear Regression
linear_model = LinearRegression()

# Huấn luyện mô hình
linear_model.fit(X_train, y_train)

# Dự đoán với tập kiểm tra
y_pred_linear = linear_model.predict(X_test)

# Tính toán MSE cho mô hình Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
print("Mean Squared Error (Linear Regression):", mse_linear)

# Xây dựng mô hình Fully Connected Neural Network (Dense)
dense_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Đầu ra dự đoán giá nhà
])

# Compile mô hình
dense_model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình với 100 epochs
dense_model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test), verbose=0)

# Dự đoán với tập kiểm tra
y_pred_dense = dense_model.predict(X_test_scaled)

# Tính toán MSE cho mô hình Fully Connected Neural Network
mse_dense = mean_squared_error(y_test, y_pred_dense)
print("Mean Squared Error (Dense Neural Network):", mse_dense)
