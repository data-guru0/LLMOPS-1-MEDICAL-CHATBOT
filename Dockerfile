# Use official Streamlit base image or Python
FROM python:3.10

# Set the working directory
WORKDIR /app

# Declare and use Hugging Face Token as build-time and run-time environment variable
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Copy all files including .env if needed
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "application.py", "--server.port=8501", "--server.address=0.0.0.0"]
