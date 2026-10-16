import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.rnn import SimpleRNN
from src.config import DATA_FILE, DEVICE, EMBEDDING_DIM, EPOCHS, HIDDEN_DIM, LEARNING_RATE
from src.tokenizer import text_to_indices
from src.utils import build_vocab, save_model


class QADataset(Dataset):

    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        question = text_to_indices(self.df.iloc[idx]["question"], self.vocab)
        answer = text_to_indices(self.df.iloc[idx]["answer"], self.vocab)
        return torch.tensor(question), torch.tensor(answer)


def train():
    df = pd.read_csv(DATA_FILE)
    vocab = build_vocab(df)

    model = SimpleRNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # batch_size is fixed at 1 because the loop below relies on it (see ans[0])
    dataloader = DataLoader(QADataset(df, vocab), batch_size=1, shuffle=True)

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for ques, ans in dataloader:
            ques = ques.to(DEVICE)
            ans = ans.to(DEVICE)

            ans = ans[0].long()

            out = model(ques)
            loss = loss_fn(out, ans)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= len(dataloader)
        print(f"Epoch : {epoch + 1}/{EPOCHS} - Loss : {total_loss:.4f}")

    save_model(model, vocab)


if __name__ == "__main__":
    train()