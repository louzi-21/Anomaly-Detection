FROM python:3.11-slim

RUN apt-get update && apt-get install -y default-mysql-client && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]

COPY start_streamlit.sh /app/

RUN chmod +x /app/start_streamlit.sh
