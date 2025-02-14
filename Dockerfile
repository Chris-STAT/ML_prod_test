# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /opt/ml/code

# Copy the current directory contents into the container at /opt/ml/code
COPY . /opt/ml/code

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages needed for SageMaker
RUN pip install --no-cache-dir sagemaker-training sagemaker-inference

# Make port 8080 available to the world outside this container (SageMaker uses 8080 by default)
EXPOSE 8080

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM train.py
ENV SAGEMAKER_SERVING_MODULE inference:handler

# Entry point for training
ENTRYPOINT ["python", "train.py"]
