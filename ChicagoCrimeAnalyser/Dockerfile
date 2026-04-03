FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any needed by ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose standard Streamlit port
EXPOSE 8501

# Entry point for the Streamlit Application
CMD ["streamlit", "run", "app/🏠_Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
