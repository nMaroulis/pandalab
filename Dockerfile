# Use official Python image as base
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable output buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies needed for certain Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Upgrade pip and install required Python packages
RUN pip install --no-cache-dir --upgrade pip

# Install the required Python packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create a directory for logs
RUN mkdir -p /app/logs

# Copy the rest of the application files into the container
COPY . /app/

# Expose the necessary ports for FastAPI and Streamlit
EXPOSE 8501

CMD bash -c "nohup streamlit run frontend_router.py --server.port 8501 >> /app/logs/frontend.log 2>&1 & \
              tail -f /dev/null"
