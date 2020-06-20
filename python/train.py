import datetime

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def evaluate(model, features):
    with torch.no_grad():
        return model(features)

def eval_accuracy(model, test_data, batch_size=32, seq_length=20, input_dim=7):
    i = batch_size
    total = 0
    correct = 0
    while i < test_data.shape[0]:
        features = torch.tensor([i for i in test_data.values[i-batch_size:i,0]], dtype=torch.float32)
        targets = torch.tensor([i for i in test_data.values[i-batch_size:i,1]], dtype=torch.long)
        features = features.reshape(seq_length, batch_size, input_dim)

        output = model(features)
        pred = torch.argmax(output, dim=1)

        correct += (pred == targets).sum()
        total += targets.shape[0]
        i += batch_size
        #print("evaluating...", i)

    return float(correct/total)

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers=10):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, layers)

        self.hidden2tag = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        hidden = (torch.randn(self.layers, inputs.shape[1], self.hidden_dim, dtype=torch.float32),
                  torch.randn(self.layers, inputs.shape[1], self.hidden_dim, dtype=torch.float32))

        # seq length, batch_size, features
        lstm_out, hidden = self.lstm(inputs, hidden)
        #print(lstm_out.shape)

        tags = self.hidden2tag(lstm_out[-1,:,:].view(inputs.shape[1], -1))
        tag_scores = F.log_softmax(tags, dim=1)

        return tags

@click.command()
@click.option('--train-path', required=True)
@click.option('--test-path', required=True)
@click.option('--eval-path', required=True)
def main(train_path, test_path, eval_path):

    train_data = pd.read_parquet(train_path)
    test_data  = pd.read_parquet(test_path)
    INPUT_DIM = 7
    HIDDEN_DIM = 4
    OUTPUT_DIM = 2

    seq_length = 200
    batch_size = 32
    model = LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, batch_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 3

    for epoch in range(epochs):
        print(epoch)
        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        i = batch_size
        while i < train_data.shape[0]:
            idx = indices[i-batch_size:i]
            features = torch.tensor([i for i in train_data.values[idx,0]], dtype=torch.float32)
            targets = torch.tensor([i for i in train_data.values[idx,1]], dtype=torch.long)

            #model.zero_grad()
            optimizer.zero_grad()
            features = features.reshape(seq_length, batch_size, INPUT_DIM)

            y_0 = model(features)

            loss = loss_function(y_0, targets)
            loss.backward()
            optimizer.step()
            i += batch_size

            if i / batch_size % 50 == 0:
                accuracy = eval_accuracy(model, test_data, batch_size=batch_size, seq_length=seq_length, input_dim=INPUT_DIM)
                print(f"iter = {i}, accuracy={accuracy}, loss={loss.data}")
                #print(i, loss.data)

if __name__=="__main__":
    main()
