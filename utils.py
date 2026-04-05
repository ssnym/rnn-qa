import torch
from model import SimpleRNN
from tokenizer import tokenize


MODEL_PATH = 'config/rnn_checkpoint.pt'

def load_model():
    """
    Used in inference.predict
    Function to load model + vocabulary 
    """

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    vocab = checkpoint["vocab"]
    model = SimpleRNN(len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    return model, vocab


def save_model(model, vocab):
    """
    Used in train.train
    Function to save model + vocabulary 
    """

    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
    }, MODEL_PATH)



def build_vocab(df) -> dict:
    """
    Used in train.train
    To build vocabulary for training from data
    """

    vocab = {'<UNK>' : 0}

    for question in df['question']:
        for token in tokenize(question):
            if token not in vocab:
                vocab[token] = len(vocab)
    
    for answer in df['answer']:
        for token in tokenize(answer):
            if token not in vocab:
                vocab[token] = len(vocab)    

    return vocab