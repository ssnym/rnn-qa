from src.config import CONFIDENCE_THRESHOLD, DEFAULT_QUESTION, DEVICE, FALLBACK_ANSWER
from src.tokenizer import text_to_indices
from src.utils import load_model
import torch
from models.rnn import SimpleRNN


def predict(model: SimpleRNN, question: str, vocab: dict, device=DEVICE, threshold: float = CONFIDENCE_THRESHOLD):
    """Predict an answer and its confidence from the RNN."""
    model.to(device)
    model.eval()

    ques_num = text_to_indices(question, vocab)

    if not ques_num:
        return FALLBACK_ANSWER, 0.0

    ques_tensor = torch.tensor(ques_num).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(ques_tensor)
        probs = torch.softmax(logits, dim=1)
        value, index = torch.max(probs, dim=1)

    confidence = value.item()

    if confidence < threshold:
        return FALLBACK_ANSWER, confidence

    return list(vocab.keys())[index.item()], confidence


if __name__ == "__main__":
    model, vocab = load_model()
    out, confidence = predict(model, DEFAULT_QUESTION, vocab)
    print(out, confidence)