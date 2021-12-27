import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules import dropout
from torch.nn.modules.linear import Identity

class LSTM(nn.Module):
    def __init__(self, 
                 max_words=10000,
                 emb_size=256, 
                 hid_size=256,
                 num_layers=2,
                 drop_out=0.2,
                 max_norm=10, 
                 bidirectional=True):

        super(LSTM, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = drop_out
        self.Embedding = nn.Embedding(max_words, emb_size, max_norm=max_norm)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.dp = nn.Dropout(self.dropout)
        if bidirectional: self.fc1 = nn.Linear(self.hid_size*2, self.hid_size)
        else: self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.dp(x)
        x, _ = self.LSTM(x)
        x = self.dp(x)
        x = F.relu(self.fc1(x))
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        out = F.sigmoid(self.fc2(x))
        return out

class ResLSTM(nn.Module):
    def __init__(self, 
                 max_words=10000,
                 emb_size=256, 
                 hid_size=256,
                 num_layers=2,
                 drop_out=0.2,
                 max_norm=10, 
                 bidirectional=True):

        super(ResLSTM, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = drop_out
        self.Embedding = nn.Embedding(max_words, emb_size, max_norm=max_norm)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.dp = nn.Dropout(self.dropout)
        if bidirectional: self.fc1 = nn.Linear(self.hid_size*2, self.hid_size)
        else: self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)
        self.fc3 = nn.Linear(self.hid_size, self.hid_size*2)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.dp(x)
        out, _ = self.LSTM(x)
        out = out + self.fc3(x)
        out = self.dp(out)
        out = F.relu(self.fc1(out))
        out = F.avg_pool2d(out, (out.shape[1], 1)).squeeze()
        out = F.sigmoid(self.fc2(out))
        return out