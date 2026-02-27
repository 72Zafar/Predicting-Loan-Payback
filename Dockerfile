# USE official Docker image for Python 3.10
FROM python:3.10-slim-bullseye

# Set the working directory 
WORKDIR /app

# Install system dependencies required by lightgbm and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Code your application
COPY . /app

# Install dependencied
RUN pip install --upgrade pip
RUN pip install --default-timeout=200 -r requirements.txt


 
# Expose the port that the application will run on
EXPOSE 5000


# Command to run the application
CMD ["python3", "app.py"]