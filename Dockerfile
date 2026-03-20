# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install git (needed to clone RADAR if not present locally)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# If RADAR directory was not copied from local context, clone it from GitHub
RUN if [ ! -d "RADAR" ]; then \
        git clone https://github.com/ari-dasci/S-RADAR.git && \
        mv S-RADAR/RADAR ./RADAR && \
        rm -rf S-RADAR; \
    fi

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]