# Use the official CUDA 12.6 base image
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Install Python and ffmpeg and required dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python3 as the default Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the project files into the container
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to execute the script
CMD ["python", "-u", "main.py"]