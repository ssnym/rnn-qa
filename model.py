# import torch
import torch.nn as nn

MODEL_VERSION = '1.0.0'

# RNN architecture
class SimpleRNN(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim=50)
        self.rnn = nn.RNN(50, 64, batch_first = True)
        self.fcnn = nn.Linear(64, vocab_size)

    def forward(self, question):
        x = self.embeddings(question)
        _, hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        out = self.fcnn(hidden)
        return out