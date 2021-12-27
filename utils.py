import numpy as np
import torch.nn as nn
import yaml
import torch
from torchstat import stat

# drop the most/least frequent words
def dataPrepocess(trainData, testData, end_idx = 10000):
    idx = np.where(trainData >= end_idx)
    trainData[idx] = 0

    idx = np.where(testData >= end_idx)
    testData[idx] = 0
        
    return trainData, testData

# init weight
def weight_init(model):
    for layer in model.modules():
        if type(layer) in [nn.Linear]:
            nn.init.xavier_normal_(layer.weight)

# get hyperparameter config
def getConfig(filename):
    with open(filename, 'r') as f:
        cfg = yaml.load(f)

    return cfg

def saveConfig(cfg, filename):
    with open(filename, 'w') as f:
        yaml.dump(cfg, f)
    return