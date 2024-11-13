from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from unidecode import unidecode
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

API_KEY = "716271569df24ee8a11100903242910"
BASE_URL = "http://api.weatherapi.com/v1/current.json"

city_aliases = {
    "hanoi": "ha noi",
    "hochiminh": "ho chi minh",
    "danang": "da nang",
    # Bổ sung các tên thành phố khác nếu cần
}
def normalize_city_name(city):
    # Loại bỏ dấu tiếng Việt, chuyển sang chữ thường và loại bỏ khoảng trắng thừa
    city = unidecode(city).lower().strip()
    # Thay thế tên thành phố bằng dạng chuẩn nếu có trong danh sách alias
    city = city_aliases.get(city, city)
    return city

def get_weather_data(city):
    response = requests.get(f"{BASE_URL}?key={API_KEY}&q={city}&aqi=no")
    if response.status_code == 200:
        return response.json()
    return None


@app.route('/')
def index():
    # Không trả về thông tin thời tiết ban đầu để giao diện chỉ hiển thị dòng chữ "Trang dự báo thời tiết"
    return render_template('index.html', current_weather=None)

@app.route('/weather', methods=['POST'])
def get_weather():
    city = request.form.get('city')
    city = normalize_city_name(city)
    weather_data = get_weather_data(city)
    if weather_data:
        current_date = datetime.strptime(weather_data['location']['localtime'], "%Y-%m-%d %H:%M").date()
        current_weather = {
            'location': weather_data['location']['name'],
            'temp_c': weather_data['current']['temp_c'],
            'condition': weather_data['current']['condition']['text'],
            'icon': weather_data['current']['condition']['icon'],
            'lat': weather_data['location']['lat'],
            'lon': weather_data['location']['lon'],
            'date': current_date.strftime("%d/%m/%Y")
        }
        return jsonify(current_weather)
    else:
        return jsonify({'error': 'City not found!'}), 404


def predict_weather(data, start_date=None):
    if start_date is None:
        start_date = datetime.now().strftime('%Y-%m-%d')
    
    # Danh sách các cột (features) cần thiết
    features = ['tempmax', 'tempmin', 'humidity', 'feelslikemax', 'feelslikemin',
                'dew', 'precip', 'precipprob', 'snow', 'snowdepth', 'windgust',
                'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                'solarradiation', 'solarenergy', 'uvindex', 'moonphase']
    
    target_tempmax = 'tempmax'
    target_tempmin = 'tempmin'
    target_precip_type = 'preciptype'

    # Kiểm tra xem các cột cần thiết có tồn tại trong dữ liệu hay không
    if not all(col in data.columns for col in features + [target_tempmax, target_tempmin, target_precip_type, 'datetime']):
        print("Data is insufficient for prediction.")
        return []
    
    # Xử lý giá trị NaN cho cột preciptype
    data['preciptype'] = data['preciptype'].fillna('sun')
    data = data.ffill().bfill()

    # Chuyển đổi cột 'datetime' thành kiểu ngày tháng
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data.dropna(subset=['datetime'])

    # Huấn luyện mô hình dự đoán nhiệt độ
    model_tempmax = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    model_tempmin = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)

    X = data[features]
    y_tempmax = data[target_tempmax]
    y_tempmin = data[target_tempmin]

    model_tempmax.fit(X, y_tempmax)
    model_tempmin.fit(X, y_tempmin)

    # Huấn luyện mô hình dự đoán loại mưa
    model_precip = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=42, class_weight='balanced')

    model_precip.fit(X, data["preciptype"])

    predictions = []
    last_row = data.iloc[-1]
    start_date = pd.to_datetime(start_date)

    # Dự đoán thời tiết cho 7 ngày tiếp theo
    for i in range(1, 8):
        next_date = start_date + pd.Timedelta(days=i)
        
        # Tạo new_data dựa trên các cột features
        new_data = pd.DataFrame({
        'tempmax': [last_row['tempmax'] + np.random.uniform(-2, 2)],
        'tempmin': [last_row['tempmin'] + np.random.uniform(-2, 2)],
        'humidity': [last_row['humidity'] + np.random.uniform(-10, 10)],
        'feelslikemax': [last_row['feelslikemax'] + np.random.uniform(-2, 2)],
        'feelslikemin': [last_row['feelslikemin'] + np.random.uniform(-2, 2)],
        'dew': [last_row['dew'] + np.random.uniform(-1, 1)],
        'precip': [last_row['precip'] + np.random.uniform(-0.5, 0.5)],
        'precipprob': [last_row['precipprob'] + np.random.uniform(-5, 5)],
        'snow': [last_row['snow'] + np.random.uniform(-0.5, 0.5)],
        'snowdepth': [last_row['snowdepth'] + np.random.uniform(-0.5, 0.5)],
        'windgust': [last_row['windgust'] + np.random.uniform(-1, 1)],
        'windspeed': [last_row['windspeed'] + np.random.uniform(-2, 2)],
        'winddir': [last_row['winddir'] + np.random.uniform(-5, 5)],
        'sealevelpressure': [last_row['sealevelpressure'] + np.random.uniform(-1, 1)],
        'cloudcover': [last_row['cloudcover'] + np.random.uniform(-5, 5)],
        'visibility': [last_row['visibility'] + np.random.uniform(-1, 1)],
        'solarradiation': [last_row['solarradiation'] + np.random.uniform(-10, 10)],
        'solarenergy': [last_row['solarenergy'] + np.random.uniform(-10, 10)],
        'uvindex': [last_row['uvindex'] + np.random.uniform(-1, 1)],
        'moonphase': [last_row['moonphase'] + np.random.uniform(-0.1, 0.1)]
    })

        # Thêm các cột còn thiếu với giá trị mặc định từ last_row
        for col in features:
            if col not in new_data.columns:
                new_data[col] = last_row[col]

        # Sắp xếp lại thứ tự cột để khớp với mô hình
        new_data = new_data[features]


        # Dự đoán nhiệt độ
        predicted_tempmax = model_tempmax.predict(new_data)[0]
        predicted_tempmin = model_tempmin.predict(new_data)[0]
        predicted_temp = (predicted_tempmax + predicted_tempmin) / 2

        # Dự đoán loại mưa
        predicted_precip_type = model_precip.predict(new_data)[0]

        predictions.append({
            'date': next_date.strftime('%Y-%m-%d'),
            'tempmax': predicted_tempmax,
            'tempmin': predicted_tempmin,
            'temp': predicted_temp,
            'precip_type': predicted_precip_type
        })
        
    print("OOB score of tempmax: ", model_tempmax.oob_score_)
    print("OOB score of tempmin: ", model_tempmin.oob_score_)
    print("OOB score of precip_type: ", model_precip.oob_score_)

    # Đánh giá mô hình hồi quy
    y_pred_tempmax = model_tempmax.predict(X)
    print("MSE for tempmax:", mean_squared_error(y_tempmax, y_pred_tempmax))
    y_pred_tempmin = model_tempmin.predict(X)
    print("MSE for tempmax:", mean_squared_error(y_tempmin, y_pred_tempmin))
    # Đánh giá mô hình phân loại
    y_pred_precip = model_precip.predict(X)
    print("Accuracy for preciptype:", accuracy_score(data["preciptype"], y_pred_precip))
    print("Confusion Matrix for preciptype:\n", confusion_matrix(data["preciptype"], y_pred_precip))
        
    return predictions



@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json  # Hoặc request.form nếu bạn gửi dữ liệu dưới dạng form
    city = data.get('city')  # Hoặc data['city'] nếu bạn chắc chắn nó sẽ có giá trị
    city = normalize_city_name(city)
    if not city:
        return jsonify({"error": "City is required!"}), 400

    file_path = f"{city.replace(' ', '_')}.csv"

    try:
        data = pd.read_csv(file_path)
        predictions = predict_weather(data)
        return jsonify(predictions)
    except FileNotFoundError:
        return jsonify({'error': 'Data file not found for this city.'}), 404


def get_coordinates(city):
    response = requests.get(f"{BASE_URL}?key={API_KEY}&q={city}&aqi=no")
    if response.status_code == 200:
        data = response.json()
        return data['location']['lon'], data['location']['lat']
    return None, None
def plot_weather(data):
    dates = data['datetime']
    tempmax = data['tempmax']
    tempmin = data['tempmin']
    temp = data['temp']  # Sử dụng temp thay vì humidity
    
    plt.figure(figsize=(10, 6))
    
    # Vẽ các đường biểu diễn tempmax, tempmin và temp
    plt.plot(dates, tempmax, label='Temp Max', color='red', marker='o')
    plt.plot(dates, tempmin, label='Temp Min', color='blue', marker='o')
    plt.plot(dates, temp, label='Temp', color='green', marker='o')  # Sửa đây
    
    # Thêm tiêu đề và nhãn
    plt.title('Weather Data: Temp Max, Temp Min, and Temp')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend()

    # Lưu biểu đồ vào bộ nhớ dưới dạng hình ảnh PNG
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    img_buf.close()

    return img_base64

@app.route('/plot_weather', methods=['POST'])
def plot_weather_data():
    data = request.json  # Dữ liệu thời tiết sẽ được truyền dưới dạng JSON
    
    # Giả sử data có cấu trúc giống như trong phần dự đoán thời tiết
    # Vậy chúng ta chuyển dữ liệu thành DataFrame để dễ dàng xử lý
    try:
        df = pd.DataFrame(data)
        img_base64 = plot_weather(df)
        return jsonify({'image': img_base64})
    except Exception as e:
        return jsonify({'error': f"Error plotting weather data: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True)
