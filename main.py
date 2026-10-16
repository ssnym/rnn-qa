from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import DEVICE, MODEL_VERSION
from src.inference import predict
from src.utils import load_model

model = None
vocab = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocab
    model, vocab = load_model()
    model.to(DEVICE)
    yield


app = FastAPI(lifespan=lifespan)


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
        ans, confidence = predict(model, question.question, vocab, DEVICE)
        return {
            "answer": ans.capitalize(),
            "confidence": round(confidence, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))