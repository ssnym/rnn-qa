# Train
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from utils import load_model, save_model, build_vocab
from model import SimpleRNN
from tokenizer import  text_to_indices

DATA_FILE = 'data/rnn-text-qa-data.csv'
MODEL_PATH = 'config/rnn_checkpoint.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class QADataset(Dataset):

    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        question = text_to_indices( self.df.iloc[idx]['question'] , self.vocab)
        answer = text_to_indices( self.df.iloc[idx]['answer'] , self.vocab)
        return torch.tensor(question) , torch.tensor(answer)
    
def train():
    
    df = pd.read_csv(DATA_FILE)
    vocab = build_vocab(df)

    model = SimpleRNN(len(vocab))
    model.to(device)

    learning_rate = 0.001
    epochs = 20

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = QADataset(df, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)    # since batch_size = 1, padding = no required


    for epoch in range(epochs):

        total_loss = 0
        model.train()
        for ques, ans in dataloader:

            ques = ques.to(device)
            ans = ans.to(device)

            # ans = ans.view(-1).long()
            ans = ans[0].long()

            out = model(ques)
            loss = loss_fn(out, ans)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        total_loss /= len(dataloader)
        print(f"Epoch : {epoch + 1}/{epochs} - Loss : {total_loss:.4f}")


    save_model(model, vocab)

if __name__ == "__main__":

    train()