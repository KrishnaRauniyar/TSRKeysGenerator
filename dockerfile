# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /tsrkeygeneration

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY tsrkeygeneration.py .

# Set the entry point for the container
ENTRYPOINT ["python", "tsrkeygeneration.py"]

