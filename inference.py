import torch 

from tokenizer import  text_to_indices
from model import SimpleRNN
from utils import load_model

MODEL_PATH = './config/rnn_checkpoint.pt'

# In production : get from ML-Flow
from model import MODEL_VERSION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model: SimpleRNN, question: str, vocab: dict, device, threshold=0.40):
    """Function to make prediction from a RNN """

    model.to(device)
    model.eval()

    ques_num = text_to_indices(question, vocab)

    if not ques_num:
       return "I don't Know", 0.0
    
    ques_tensor = torch.tensor(ques_num).unsqueeze(0)
    ques_tensor = ques_tensor.to(device)

    with torch.no_grad():
        logits = model(ques_tensor)
        probs = torch.softmax(logits, dim=1)
        value, index = torch.max(probs, dim = 1)

    confidence = value.item()

    if value.item() < threshold:
        return "I don't Know", confidence
    
    return list(vocab.keys())[index.item()], confidence


if __name__ == "__main__":
    
    model, vocab = load_model()
    out, confidence = predict(model,'What is the capital of France ?', vocab, device)
    print(out, confidence)
