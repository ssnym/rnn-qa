import torch

from models.rnn import SimpleRNN
from src.config import EMBEDDING_DIM, HIDDEN_DIM, MODEL_PATH, UNK_TOKEN
from src.tokenizer import tokenize


def load_model():
    """Load model and vocabulary from the checkpoint at MODEL_PATH."""
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    vocab = checkpoint["vocab"]
    model = SimpleRNN(
        vocab_size=len(vocab),
        embedding_dim=checkpoint.get("embedding_dim", EMBEDDING_DIM),
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, vocab


def save_model(model, vocab):
    """Save model weights, vocabulary, and architecture sizes to MODEL_PATH."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "embedding_dim": model.embeddings.embedding_dim,
            "hidden_dim": model.rnn.hidden_size,
        },
        MODEL_PATH,
    )


def build_vocab(df) -> dict:
    """Build a vocabulary from all questions and answers in the dataset."""
    vocab = {UNK_TOKEN: 0}

    for text in list(df["question"]) + list(df["answer"]):
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab