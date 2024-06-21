FROM python:3.9-slim

RUN  apt-get update \
    && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --default-timeout=1000 --no-cache-dir

# Copy the rest of the application code
COPY . .

# Download bart-multinews.keras model into ./bart-multinews.keras
RUN wget https://huggingface.co/mnabielap/bart-multinews/resolve/main/bart-multinews.keras -O ./bart-multinews.keras

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
