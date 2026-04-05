## RNN Question Answering API  

A simple **RNN-based Question Answering system** built using **Pytorch + FastAPI**, with Docker support

### Workflow
- Train model using `train.py`
- Model + vocab stored in `config/`
- API loads model and serves prediction  

### Project Structure

rnn-qa/  
│  
├── config/     # model artifacts    
├── data/       # sample-dataset  
│  
├── .gitignore  
├── .dockerignore  
│  
├── Dockerfile      # continer setup   
│  
├── inference.py    # prediction logic  
├── main.py     # FastAPI app / entrypoint  
├── model.py    # RNN model architecture  
│  
├── requirements.txt   
│  
├── tokenizer.py       # text processing  
├── train.py           # training script  
└── utils.py           # helper functions  

### Running Locally


```bash
# Start API  

uvicorn main:app --reload
```

```bash
# Base URL  

http://localhost:8080
```

```bash
# Example Request  

curl --location 'localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "question": "What is the capital of France"
}'
```

```bash
# Example Response   

{
    "answer": "Paris",
    "confidence": 0.85
}
```

## Docker Image
Prebuilt Docker image available with 600+ pulls  on Docker Hub 
 [https://hub.docker.com/r/ssnym/rnn-qa](https://hub.docker.com/r/ssnym/rnn-qa)

## Acknowledgements
Tutorial Followed : FastAPI course by CampusX

