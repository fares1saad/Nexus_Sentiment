# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run train.py (assuming it trains the model)
RUN python train.py

# Command to run the Flask application
CMD ["python", "api_alike.py"]