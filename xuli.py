import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO logs, 2 = WARNING logs, 3 = ERROR logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt thông báo oneDNN

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

def xuli_data():
    global df
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('data_HN.csv')

    # Điền giá trị cho các cột thiếu
    df['prcp'] = df['prcp'].fillna(df['prcp'].mean())
    df['wdir'] = df['wdir'].ffill().bfill()
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
    # print("Số lượng giá trị thiếu sau khi xử lý:\n", df.isnull().sum())

def bieudo():
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='date', y='tavg', data=df)
    plt.title('Temperature Trend')
    plt.xlabel('2015 to Now')
    plt.ylabel('Average Temperature')
    plt.show()

def tranning_LR():
    X = df[['tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres']]
    y = df['tavg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE của Linear Regression: {mse}')
    return model, mse

def tranning_RFR():
    X = df.drop(columns=['tavg', 'date'])
    y = df['tavg']

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_mse = mean_squared_error(y_test, rf_predictions)
    print(f'MSE của Random Forest: {rf_mse}')
    return rf_model, rf_mse

def tranning_LSTM():
    data = df[['tavg']].values

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lstm_model = Sequential()
    lstm_model.add(Input(shape=(X_train.shape[1], 1)))
    lstm_model.add(LSTM(50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    lstm_predictions = lstm_model.predict(X_test)

    lstm_mse = mean_squared_error(y_test, lstm_predictions)
    print(f'MSE của LSTM: {lstm_mse}')
    return lstm_model, lstm_mse

def predict_future_weather(model, last_data, model_type):
    predictions = []
    if model_type == 'LR' or model_type == 'RF':
        for _ in range(7):
            pred = model.predict([last_data])
            predictions.append(pred[0])
            last_data[0] = pred[0]  # Cập nhật tavg
    else:  # LSTM
        time_steps = 10
        last_sequence = last_data[-time_steps:].reshape(1, time_steps, 1)
        for _ in range(7):
            prediction = model.predict(last_sequence)
            predictions.append(prediction[0, 0])
            prediction = np.array(prediction).reshape(1, 1, 1)
            last_sequence = np.append(last_sequence[:, 1:, :], prediction, axis=1)

    return np.array(predictions)

if __name__ == "__main__":
    xuli_data()  # Xử lý dữ liệu
    # bieudo()  # Vẽ biểu đồ

    # Huấn luyện mô hình
    lr_model, lr_mse = tranning_LR()
    rf_model, rf_mse = tranning_RFR()
    lstm_model, lstm_mse = tranning_LSTM()

    # So sánh MSE và đưa ra phương án tối ưu
    mse_dict = {'Linear Regression': lr_mse, 'Random Forest': rf_mse, 'LSTM': lstm_mse}
    best_model_name = min(mse_dict, key=mse_dict.get)
    best_model_mse = mse_dict[best_model_name]

    print(f'Model tối ưu nhất: {best_model_name} với MSE: {best_model_mse}')

    # Dự đoán tương lai
    last_data = [df['tmin'].iloc[-1], df['tmax'].iloc[-1], df['prcp'].iloc[-1],
                 df['snow'].iloc[-1], df['wdir'].iloc[-1], df['wspd'].iloc[-1],
                 df['wpgt'].iloc[-1], df['pres'].iloc[-1]]  # Dữ liệu cuối cùng

    if best_model_name == 'Linear Regression':
        lr_predictions = predict_future_weather(lr_model, last_data, 'LR')
        print("Dự đoán thời tiết trong 1 tuần tới với Linear Regression:")
    elif best_model_name == 'Random Forest':
        rf_predictions = predict_future_weather(rf_model, last_data, 'RF')
        print("Dự đoán thời tiết trong 1 tuần tới với Random Forest:")
    else:
        lstm_predictions = predict_future_weather(lstm_model, df[['tavg']].values, 'LSTM')
        print("Dự đoán thời tiết trong 1 tuần tới với LSTM:")

    today = datetime.now()
    for i, temp in enumerate(lr_predictions if best_model_name == 'Linear Regression' else
                            rf_predictions if best_model_name == 'Random Forest' else
                            lstm_predictions):
        date_prediction = today + timedelta(days=i + 1)
        print(f'{date_prediction.strftime("%Y-%m-%d")}: {temp:.2f} °C')
