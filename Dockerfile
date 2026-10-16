# base image
FROM python:3.12-slim

# keep Python output unbuffered and avoid .pyc files in the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# working directory
WORKDIR /app

# install dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code and model artifacts
COPY main.py .
COPY models/ models/
COPY src/ src/
COPY config/ config/

# expose port
EXPOSE 8000

# command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]