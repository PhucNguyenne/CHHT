import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def xuli_data():
    global df
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('data_HCM.csv')

    # Điền giá trị cho các cột thiếu
    df['prcp'] = df['prcp'].fillna(df['prcp'].mean())
    df['wdir'] = df['wdir'].ffill().bfill()  # Điền từ trên xuống và dưới lên
    df['pres'] = df['pres'].fillna(df['pres'].mean())
    df['tavg'] = df['tavg'].fillna(df['tavg'].mean())
    df['tmin'] = df['tmin'].fillna(df['tmin'].mean())
    df['tmax'] = df['tmax'].fillna(df['tmax'].mean())
    df['wspd'] = df['wspd'].fillna(df['wspd'].mean())
    df['snow'] = df['snow'].fillna(0)
    df['wpgt'] = df['wpgt'].fillna(0)
    df['tsun'] = df['tsun'].fillna(0)

    # Chuyển đổi cột 'date' thành kiểu datetime
    df['date'] = pd.to_datetime(df['date'])

    # Kiểm tra lại số lượng giá trị thiếu
    print("Số lượng giá trị thiếu sau khi xử lý:\n", df.isnull().sum())

def tranning_LR():
    # Chọn các đặc trưng (features) và nhãn (target)
    X = df[['tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres']]
    y = df['tavg']

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE of Linear regression: {mse}')
    print(f'R^2 Score: {r2}')
    return model, mse  # Trả về mô hình và giá trị MSE

def tranning_RFR():
    # Chọn các đặc trưng (features) và nhãn (target)
    X = df[['tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres']]
    y = df['tavg']

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)

    print(f'MSE of Random Forest: {mse}')
    return model, mse  # Trả về mô hình và giá trị MSE

def tranning_LSTM():
    # Chọn dữ liệu cho LSTM
    data = df[['tavg']].values

    # Chuyển đổi dữ liệu cho LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X, y = create_dataset(data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Chia dữ liệu cho LSTM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Xây dựng mô hình LSTM
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(X_train.shape[1], 1)))  # Sử dụng lớp Input
    lstm_model.add(LSTM(50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))

    # Biên dịch mô hình
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # Huấn luyện mô hình
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)  # Ẩn thông tin huấn luyện

    # Dự đoán
    lstm_predictions = lstm_model.predict(X_test)

    # Đánh giá mô hình
    lstm_mse = mean_squared_error(y_test, lstm_predictions)
    print(f'MSE of LSTM: {lstm_mse}')
    return lstm_model, lstm_mse  # Trả về mô hình và MSE

def predict_future_weather(model, input_sequence):
    future_weather = []

    # Đảm bảo input_sequence là mảng 2D
    if isinstance(input_sequence, pd.DataFrame):
        input_sequence = input_sequence.values  # Chuyển đổi DataFrame thành mảng numpy

    last_sequence = input_sequence[-1:]  # Lấy phần tử cuối cùng

    for _ in range(7):  # Dự đoán cho 7 ngày
        # Dự đoán
        prediction = model.predict(last_sequence)  # Dự đoán với mô hình

        future_weather.append(prediction[0])  # Thêm giá trị dự đoán vào danh sách

        # Cập nhật last_sequence cho dự đoán tiếp theo
        if isinstance(model, Sequential):  # Nếu là mô hình LSTM
            last_sequence = np.append(last_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)  # Giữ lại chiều 3D cho LSTM
        else:  # Nếu là mô hình Linear Regression hoặc Random Forest
            # Kiểm tra xem last_sequence có phải là mảng 2D hay không
            if last_sequence.ndim == 1:
                last_sequence = last_sequence.reshape(1, -1)  # Chuyển thành 2D nếu cần
            last_sequence = np.append(last_sequence[:, 1:], prediction.reshape(1, -1), axis=1)  # Giữ lại chiều 2D cho LR và RF

    return future_weather


# Sử dụng hàm dự đoán trong hàm chính
if __name__ == "__main__":
    xuli_data()  # Xử lý dữ liệu trước khi huấn luyện

    # Huấn luyện các mô hình và lưu MSE
    print("Training Linear Regression...")
    lr_model, mse_lr = tranning_LR()  # Lưu cả mô hình và MSE

    print("Training Random Forest Regressor...")
    rf_model, rf_mse = tranning_RFR()  # Lưu cả mô hình và MSE

    print("Training LSTM...")
    lstm_model, lstm_mse = tranning_LSTM()  # Mô hình LSTM

    # So sánh MSE của các mô hình
    print(f'\nMSE Comparison:')
    print(f'Linear Regression: {mse_lr}')
    print(f'Random Forest: {rf_mse}')
    print(f'LSTM: {lstm_mse}')

    # Chọn mô hình với MSE thấp nhất để dự đoán
    best_model = None
    if mse_lr < rf_mse and mse_lr < lstm_mse:
        best_model = lr_model  # Chọn mô hình Linear Regression
    elif rf_mse < lstm_mse:
        best_model = rf_model  # Chọn mô hình Random Forest
    else:
        best_model = lstm_model  # Chọn mô hình LSTM

    print(f'Best Model for Prediction: {best_model}')

    # Dự báo thời tiết cho 1 tuần tới
    # Dự báo thời tiết cho 1 tuần tới
    if isinstance(best_model, Sequential):
        future_weather = predict_future_weather(best_model, df[['tavg']].values.reshape(-1, 1, 1))
    else:
        # Chọn dữ liệu với tên đặc trưng
        feature_data = df[['tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres']]
        future_weather = predict_future_weather(best_model, feature_data)
