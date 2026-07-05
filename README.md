# RNN Question Answering API  

A simple **RNN-based Question Answering API** built using **Pytorch + FastAPI** and **Docker**.  
The model is trained using a Recurrent Neural Network (RNN) and exposed through a REST API.

## Project Workflow
- Train model using `train.py`
- Model + vocab stored in `config/`
- API loads model and serves prediction  

## Project Structure

```text
rnn-qa/  
│  
├── config/     # model artifacts    
├── data/       # sample dataset  
│  
├── .gitignore  
├── .dockerignore  
│  
├── Dockerfile      # container setup   
│  
├── inference.py    # prediction logic  
├── main.py         # FastAPI app / entrypoint  
├── model.py        # RNN model architecture  
│  
├── requirements.txt   
│  
├── tokenizer.py       # text processing  
├── train.py           # training script  
└── utils.py           # helper functions  
```

## Running Locally

```bash
# Start API  
git clone https://github.com/ssnym/rnn-qa.git

cd rnn-qa

pip install -r requirements.txt

uvicorn main:app --reload
```

Base URL: `http://127.0.0.1:8000`

## Running with Docker

A prebuilt Docker image is available on Docker Hub with **650+ pulls**.

Docker Hub: [`https://hub.docker.com/r/ssnym/rnn-qa`](https://hub.docker.com/r/ssnym/rnn-qa)

```bash
# Pull the image
docker pull ssnym/rnn-qa

# Run the container:
docker run -p 8000:8000 ssnym/rnn-qa
```

The API will be available at:

```
http://127.0.0.1:8000
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

## Acknowledgements
Tutorial Followed : FastAPI course by CampusX

