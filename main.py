from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
from utils import load_model
from inference import predict
from model import MODEL_VERSION
import torch

app = FastAPI()

model = None
vocab = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_model_once():
    global model, vocab
    model, vocab = load_model()
    model.to(device)


class Question(BaseModel):
    question: Annotated[str, Field(..., description="Question asked by user")]


class Answer(BaseModel):
    answer: Annotated[str, Field(..., description="Answer by Model")]
    confidence: Annotated[
        float, Field(..., description="Confidence score for that answer")
    ]


@app.get("/")
def home():
    return {"message": "RNN Question Answering system ?"}


@app.get("/health")
def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "OK",
            "version": MODEL_VERSION,
            "model_loaded": model is not None,
        },
    )


@app.post("/predict", response_model=Answer, status_code=200)
def predict_api(question: Question):

    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Empty Question")

    try:
        ans, confidence = predict(model, question.question, vocab, device)
        ans = ans.capitalize()
        return {
            "answer": ans, 
            "confidence": round(confidence,2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
