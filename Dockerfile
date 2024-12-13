# Use official Python image from Docker Hub
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app and other necessary files to the container
COPY . /app/

# Expose the port Flask will run on
EXPOSE 8080

# Command to run the app
CMD ["python", "app.py"]
