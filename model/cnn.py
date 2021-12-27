import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,
                 max_words=10000,
                 emb_size=128,
                 max_norm=1,
                 maxlen=256,
                 out_size=128,
                 filter_heights=[5,7,11],
                 dp=0.2):
        super(CNN, self).__init__()
        self.maxlen = maxlen
        self.emb_size = emb_size

        self.Embedding = nn.Embedding(max_words, emb_size, max_norm=max_norm)
        self.conv1 = nn.ModuleList(nn.Conv2d(emb_size, out_size, (fh, 1), padding=((fh-1)//2, 0)) for fh in filter_heights)
        self.dropout = nn.Dropout(dp)
        linear_size = out_size * len(filter_heights)
        self.fc = nn.Linear(linear_size, 2)
        

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.Embedding(x)
        x = x.view(batchsize, self.maxlen, self.emb_size, 1)
        x = x.permute(0, 2, 1, 3)
        final_x = []
        for i in range(0, len(self.conv1)):
            result = F.relu(self.conv1[i](x)).squeeze(3)
            final_x.append(result)
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in final_x]
        out = torch.cat(out, dim=1)
        out = self.fc(self.dropout(out))
        return out

class Seq_CNN(nn.Module):
    def __init__(self, 
                 out_ch = 256, 
                 filter_sizes=[5,10,15,20],
                 maxlen=256,
                 dropout=0.2, 
                 emb_size=256, 
                 max_words=10000):
        super(Seq_CNN, self).__init__() 
        self.dropout_rate = dropout
        self.embedding_size = emb_size
        self.max_words = max_words
        self.out_ch = out_ch
        filter_heights = filter_sizes

        self.conv = nn.ModuleList(
            [nn.Conv2d(max_words, self.out_ch, (fh, self.embedding_size), padding=(fh - 1, 0)) for fh in
             filter_heights])

        linear_size = self.out_ch * len(filter_heights)


        self.lrn = nn.LocalResponseNorm(1, 1, 1 / 2, 1)  # for the normalization of (1 +z^2)^(-1/2)
        self.linear = nn.Linear(linear_size, 2)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, input):
        """
        :param input: word embedding of words
        """
        batchsize, words_len = input.size()[0], input.size()[1]
        one_hot_emb = F.one_hot(input, num_classes=self.max_words).float()
        one_hot_emb = one_hot_emb.view(batchsize, words_len, 1, self.max_words)
        one_hot_emb = one_hot_emb.permute(0, 3, 2, 1)
        final_x = []
        for i in range(0, len(self.conv)):
            result = F.relu(self.conv[i](one_hot_emb)).squeeze(3)
            final_x.append(result)

        x = [self.lrn(F.max_pool1d(i, i.size(2))).squeeze(2) for i in final_x]
        x = torch.cat(x, dim=1)

        final = self.linear(self.dropout(x))

        return final

if __name__=='__main__':
    model = CNN().cuda()
    x = torch.randint(0,10000,(256,256)).cuda()
    y = model(x)