import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
from torch.nn.modules import dropout

## ! unfinished 
class Transformers(nn.Module):
    def __init__(self,
                 maxlen=256,
                 embed_dim=256,
                 nhead=8,
                 num_layers=6,
                 max_words=10000,
                 dim_feedforward=2048,
                 dp=0.1,
                 max_norm=1,
                 activation='gelu'):
        super(Transformers, self).__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.parameter.Parameter(torch.randn(1, embed_dim))
        self.Embedding = nn.Embedding(max_words, embed_dim, max_norm=max_norm)
        self.pos_emb = PositionalEncoder(max_len=maxlen+1, d_model=embed_dim, dropout=dp)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                        dim_feedforward=dim_feedforward,
                                                        nhead=nhead, 
                                                        dropout=dp,
                                                        activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(embed_dim, embed_dim*2)
        self.fc2 = nn.Linear(embed_dim*2, 2)

        

    def forward(self, x):
        x = self.Embedding(x)
        x = torch.cat((x, self.cls_token.repeat(x.size()[0], 1).view(-1, 1, self.embed_dim)), dim=1)
        x = self.pos_emb(x) # position embedding
        x = x.permute(1,0,2) # (N L E) to (L N E)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)
        x = x[:, 0] # cls token
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class PositionalEncoder(nn.Module):
    def __init__(self, d_model=256, max_len = 256, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        ## return self.dropout(x)
        return x