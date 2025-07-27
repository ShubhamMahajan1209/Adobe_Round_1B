# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures the image is smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local directories and files into the container
# 1. The source code of the application
COPY src/ /app/src/
# 2. The pre-downloaded model for offline use
COPY model_cache/ /app/model_cache/
# 3. The input data for the test case
COPY test_case/ /app/test_case/
# 4. The run script
COPY run.sh .

# Make the run script executable
RUN chmod +x run.sh

# Define the command to run the application
# This will execute run.sh when the container starts
CMD ["./run.sh"]