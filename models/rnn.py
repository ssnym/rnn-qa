import torch.nn as nn


class SimpleRNN(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fcnn = nn.Linear(hidden_dim, vocab_size)

    def forward(self, question):
        x = self.embeddings(question)
        _, hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        return self.fcnn(hidden)