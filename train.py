import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model.lstm import LSTM, ResLSTM
from model.transfomers import Transformers
from model.cnn import Seq_CNN, CNN
from utils import dataPrepocess, weight_init, getConfig, saveConfig
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log')

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='config/cnn.yaml', help='.yaml config file path')
parser.add_argument('--save', type=str, default='./cnn/', help='save folder')

opt = parser.parse_args()
print(opt)

cfg_path = opt.cfg
save = opt.save

## cfg_path = 'config/transformers.yaml'
## cfg_path = 'config/lstm.yaml'
## save = './transformers/'
## save = './lstm/'
if not os.path.isdir(save): os.makedirs(save)

# get config hyperparameters
cfg = getConfig(cfg_path)
print(cfg)

data = cfg['data']
# WARNNING: end_idx should equal or less than max_words
start_idx = data['start_idx']
end_idx = data['end_idx']
maxlen = data['maxlen']

optimizer_type = cfg['optimizer']
scheduler_cfg = cfg['scheduler']
lr = cfg['lr']
weight_decay = cfg['weight_decay']
batchsize = cfg['batchsize']
epoch = cfg['epoch']
num_worker = cfg['num_worker']

model_cfg = cfg['model']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('loading data')
# load dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(skip_top=start_idx)
# set sequence to the same length
train_data = pad_sequences(train_data, maxlen=maxlen, padding='post', truncating='post')
test_data = pad_sequences(test_data, maxlen=maxlen, padding='post', truncating='post')
train_data = np.array(train_data)
test_data = np.array(test_data)
# drop certain word idx
train_data, test_data = dataPrepocess(train_data, test_data, end_idx=end_idx)
# dataLoader
trainDataset = TensorDataset(torch.LongTensor(train_data), torch.LongTensor(train_labels))
testDataset = TensorDataset(torch.LongTensor(test_data), torch.LongTensor(test_labels))
trainLoader = DataLoader(trainDataset, batch_size=batchsize, shuffle=True, num_workers=num_worker)
testLoader = DataLoader(testDataset, batch_size=batchsize, shuffle=True, num_workers=num_worker)

# build model
print('build model')
if model_cfg['name'] == 'LSTM':
    model = ResLSTM(max_words=model_cfg['max_words'],
                 emb_size=model_cfg['emb_size'],
                 hid_size=model_cfg['hid_size'],
                 num_layers=model_cfg['num_layers'],
                 drop_out=model_cfg['drop_out'],
                 bidirectional=model_cfg['bidirectional'])
elif model_cfg['name'] == 'transformers':
    model = Transformers(maxlen=model_cfg['maxlen'],
                         embed_dim=model_cfg['embed_dim'],
                         nhead=model_cfg['nhead'],
                         num_layers=model_cfg['num_layers'],
                         max_words=model_cfg['max_words'],
                         dp=model_cfg['drop_out'],
                         dim_feedforward=model_cfg['dim_feedforward'],
                         max_norm=model_cfg['max_norm'],
                         activation=model_cfg['activation'])
    # model.cls_token = model.cls_token.to(device)
elif model_cfg['name'] == 'cnn':
    model = CNN(out_size=model_cfg['out_size'],
                filter_heights=model_cfg['filter_heights'],
                dp=model_cfg['dropout'],
                emb_size=model_cfg['emb_size'],
                max_norm=model_cfg['max_norm'],
                maxlen=maxlen,
                max_words=model_cfg['max_words'])
else:
    raise NotImplemented
model.apply(weight_init)


if optimizer_type == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    raise NotImplemented

if scheduler_cfg['name'] == 'steplr':
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=scheduler_cfg['step_size'],gamma=scheduler_cfg['gamma'])
elif scheduler_cfg['name'] == 'ReduceLROnPlateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_cfg['patience'])
else:
    raise NotImplemented

model.to(device)
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0,1,2])

if os.path.isfile(os.path.join(save, 'config.yaml')):
    best_cfg = getConfig(os.path.join(save, 'config.yaml'))
    best_acc = best_cfg['best acc']
else: best_acc = 0
print('best acc: ', best_acc)

for n in range(epoch):

    loss_per_epoch = 0
    correct = 0
    # traning
    model.train()
    for i, (x,y) in enumerate(trainLoader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        yhat = model(x)
        loss = F.cross_entropy(yhat, y)
        loss_per_epoch += loss
        loss.backward()
        optimizer.step()

        pred = torch.argmax(yhat, dim=-1)
        correct += torch.sum(pred == y).float().cpu()
    acc = correct / 25000
    loss_per_epoch = loss_per_epoch / i

    # test
    loss_val_per_epoch = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(testLoader):
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = F.cross_entropy(yhat, y)
            loss_val_per_epoch += loss
            pred = torch.argmax(yhat, dim=-1)
            correct += torch.sum(pred == y).float().cpu()
        acc_val = correct / 25000
        loss_val_per_epoch = loss_val_per_epoch / i
    if scheduler_cfg['name'] == 'ReduceLROnPlateau': scheduler.step(loss_val_per_epoch)
    else: scheduler.step()

    writer.add_scalar('train acc', acc, n)
    writer.add_scalar('train loss', loss_per_epoch, n)
    writer.add_scalar('val acc', acc_val, n)
    writer.add_scalar('val loss', loss_val_per_epoch, n)

    if best_acc < acc_val:
        best_acc = acc_val
        cfg['best acc'] = round(float(best_acc),4)
        cfg['n_epoch'] = n + 1
        torch.save(model.state_dict(), os.path.join(save, 'best.pth'))
        saveConfig(cfg, os.path.join(save, 'config.yaml'))

    print('[Epoch: %i/%i] \t train loss %.6f\t train acc %.4f\t test loss %.6f\t test acc %.4f\t best acc %.4f\t lr %.3e'\
                %(n+1, epoch, loss_per_epoch, acc, loss_val_per_epoch, acc_val, best_acc,optimizer.state_dict()['param_groups'][0]['lr']))
    
