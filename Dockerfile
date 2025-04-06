# Use official Streamlit base image or Python
FROM python:3.10

# Accept build-time argument for Hugging Face token
ARG HF_TOKEN

# Set the Hugging Face token as an environment variable inside the container
ENV HF_TOKEN=${HF_TOKEN}

# Optionally: store the token in a .env file inside the container for python-dotenv
RUN echo "HF_TOKEN=${HF_TOKEN}" > /app/.env
# Set the working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "application.py", "--server.port=8501", "--server.address=0.0.0.0"]
