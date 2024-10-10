import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime, timedelta  # Thêm thư viện datetime

def xuli_data():
    global df
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('data_HN.csv')

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

def bieudo():
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='date', y='tavg', data=df)
    plt.title('Temperature Trend')
    plt.xlabel('2015 to Now')
    plt.ylabel('Average Temperature')
    plt.show()

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
    return mse  # Trả về giá trị MSE

def tranning_RFR():
    X = df.drop(columns=['tavg', 'date'])  # Loại bỏ cột 'tavg' (nhãn) và 'date' (không cần thiết)
    y = df['tavg']

    # Tạo Imputer để thay thế NaN bằng giá trị trung bình
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Dự đoán
    rf_predictions = rf_model.predict(X_test)

    # Đánh giá mô hình
    rf_mse = mean_squared_error(y_test, rf_predictions)
    print(f'MSE of Random Forest: {rf_mse}')
    return rf_mse  # Trả về giá trị MSE

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

def predict_future_weather(model, data, time_steps=10, future_days=7):
    # Tạo tập dữ liệu cho dự đoán
    last_sequence = data[-time_steps:]  # Lấy dãy cuối cùng của dữ liệu
    predictions = []

    for _ in range(future_days):
        # Chuyển đổi dữ liệu để phù hợp với đầu vào của mô hình
        last_sequence = last_sequence.reshape(1, time_steps, 1)
        prediction = model.predict(last_sequence)
        predictions.append(prediction[0, 0])

        # Cập nhật dãy cuối cùng để bao gồm giá trị dự đoán mới
        prediction = np.array(prediction).reshape(1, 1, 1)
        last_sequence = np.append(last_sequence[:, 1:, :], prediction, axis=1)

    # Chuyển đổi kết quả thành một mảng NumPy
    return np.array(predictions)

# Sử dụng hàm dự đoán trong hàm chính
if __name__ == "__main__":
    xuli_data()  # Xử lý dữ liệu trước khi huấn luyện
    # bieudo()     # Vẽ biểu đồ

    # Huấn luyện các mô hình và lưu MSE
    # print("Training Linear Regression...")
    mse_lr = tranning_LR()  # Mô hình hồi quy tuyến tính

    # print("Training Random Forest Regressor...")
    rf_mse = tranning_RFR()  # Mô hình Random Forest
    # print(f'MSE of Random Forest: {rf_mse}')

    # print("Training LSTM...")
    lstm_model, lstm_mse = tranning_LSTM()  # Mô hình LSTM
    # print(f'MSE of LSTM: {lstm_mse}')

    # So sánh MSE của các mô hình
    print(f'\nMSE Comparison:')
    print(f'Linear Regression: {mse_lr}')
    print(f'Random Forest: {rf_mse}')
    print(f'LSTM: {lstm_mse}')

    # Dự báo thời tiết cho 1 tuần tới
    future_weather = predict_future_weather(lstm_model, df[['tavg']].values)

    # Lấy ngày hiện tại
    today = datetime.now()
    
    print("Dự đoán thời tiết trong 1 tuần tới:")
    for i, temp in enumerate(future_weather):
        date_prediction = today + timedelta(days=i + 1)  # Tính ngày dự đoán
        print(f'{date_prediction.strftime("%Y-%m-%d")}: {temp:.2f} °C')  # Hiển thị ngày và nhiệt độ
