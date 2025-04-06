# Use official Streamlit base image or Python
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy files
COPY .env .env
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "application.py", "--server.port=8501", "--server.address=0.0.0.0"]
