# Use official Python image
FROM python:3.10

WORKDIR /app

COPY . .

# Remove this line
# COPY .env .env

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "application.py", "--server.port=8501", "--server.address=0.0.0.0"]
