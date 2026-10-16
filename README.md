# RNN Question Answering API  

A simple **RNN-based Question Answering API** built using **Pytorch** and **FastAPI**
A small RNN is trained on question and answer pairs and served through a REST API, and the project is packaged as a **Docker** image.


## How It Works

- Train the model with `python -m src.train`. This saves the weights and vocabulary to `artifacts/`.
- The API loads the saved model at startup and returns an answer with a confidence score.
- When confidence falls below the threshold, the API returns `"I don't know"`.


## Project Workflow
- Train model with `python -m src.train`. This saves the weights and vocabulary to `artifacts/`.
- The API loads the saved model at startup and returns an answer with a confidence score.
- When confidence falls below the threshold, the API returns `"I don't know"`.


## Running Locally

```bash

git clone https://github.com/ssnym/rnn-qa.git
cd rnn-qa

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Create your local settings file
cp .env.example .env          # Windows (cmd): copy .env.example .env

pip install -r requirements.txt

uvicorn main:app --reload
```

Base URL: `http://127.0.0.1:8000`

API docs are available at `http://127.0.0.1:8000/docs`.


## Training

```bash
python -m src.train
```

## Running with Docker

A prebuilt Docker image is available on Docker Hub with **2.1K pulls**.

Docker Hub: [`https://hub.docker.com/r/ssnym/rnn-qa`](https://hub.docker.com/r/ssnym/rnn-qa)

```bash
docker pull ssnym/rnn-qa
docker run -p 8000:8000 ssnym/rnn-qa
```


## Example Request

```bash
curl --location 'http://127.0.0.1:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "question": "What is the capital of France"
}'
```

## Example Response

```json
{
    "answer": "Paris",
    "confidence": 0.85
}
```


## API Endpoints

### `GET /`

Returns a welcome message.

**Response**
```json
{
    "message": "RNN Question Answering system ?"
}
```

### `GET /health`

Returns the service status, model version, and whether the model is loaded.

**Response**
```json
{
    "status": "OK",
    "version": "1.0.0",
    "model_loaded": true
}
```

### `POST /predict`

Answers a question.

**Request**
```json
{
    "question": "What is the capital of France?"
}
```

**Response**
```json
{
    "answer": "Paris",
    "confidence": 1.0
}
```


## Acknowledgements
Tutorial Followed : FastAPI course by CampusX

