# base image
FROM python:3.12-slim

# working directory
WORKDIR /app

# copy requirements
COPY requirements.txt .

# run
RUN pip install --no-cache-dir -r requirements.txt

# copy
COPY main.py .
COPY inference.py .
COPY model.py .
COPY tokenizer.py .
COPY config/ config/


# expose port
EXPOSE 8000

# command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]