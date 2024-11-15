# Sử dụng Python 3.10 (có thể điều chỉnh phiên bản nếu cần)
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY . /app

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file yêu cầu và cài đặt các thư viện Python
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Thiết lập biến môi trường cho Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Expose cổng 5000 (cổng mặc định của Flask)
EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "app.py"]
