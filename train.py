from __future__ import division

import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
sys.path.append(base_dir + '/models/')
sys.path.append(base_dir + '/task/')

'''Baseline Dataset'''
from sequential_mnist import SequentialMNIST

'''Models of Interest'''
from lstm import LSTM
from irnn import IRNN
from gru import GRU
from peephole_lstm import Peephole
from ugrnn import UGRNN
from intersection_rnn import IntersectionRNN
from rnn import RNN


#Define global variables and arguments for experimentation
parser = argparse.ArgumentParser(description='Nueron Connection')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='RMSprop optimizer momentum (default: 0.9)')
parser.add_argument('--alpha', type=float, default=0.95,
                    help='RMSprop alpha (default: 0.95)')
parser.add_argument('--epochs', type=int, default=100,
                    help='num training epochs (default: 100)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--hx', type=int, default=100,
                    help='hidden vec size for lstm models (default: 100)')
parser.add_argument('--layers', type=int, default=1,
                    help='num recurrent layers (default: 1)')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--model-type', type=str, default='lstm',
                    help='rnn, lstm, gru, irnn, ugrnn, rnn+, peephole')
parser.add_argument('--task', type=str, default='seqmnist',
                    help='seqmnist, pseqmnist')
parser.add_argument('--sequence-len', type=int, default=784,
                    help='mem seq len (default: 784)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use gpu')

args = parser.parse_args()

#Check if cuda is available
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print("Using GPU Acceleration")

#Set experimentation seed
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

np.random.seed(args.seed)
random.seed(args.seed)

#Define helper function
def log_sigmoid(x):
    return torch.log(F.sigmoid(x))

#Load data loader depending on task of interest
if args.task == 'seqmnist':
    print("Loading SeqMNIST")
    dset = SequentialMNIST()
else:
    print("Loading PSeqMNIST")
    dset = SequentialMNIST(permute=True)

data_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=2, shuffle=True)
activation = nn.LogSoftmax(dim=1)
criterion = nn.CrossEntropyLoss(size_average=False, reduce=False)

#Define deep recurrent neural network
def create_model():
    if args.model_type == 'lstm':
        return LSTM(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          layers=args.layers)
    elif args.model_type == 'rnn':
        return RNN(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          layers=args.layers)
    elif args.model_type == 'irnn':
        return IRNN(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          layers=args.layers)
    elif args.model_type == 'gru':
        return GRU(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          layers=args.layers)
    elif args.model_type == 'rnn+':
        if args.layers == 1:
            args.layers = 2
        return IntersectionRNN(input_size=dset.input_dimension,
                                      hidden_size=args.hx,
                                      output_size=dset.output_dimension,
                                      layers=args.layers)
    elif args.model_type == 'peephole':
        return Peephole(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          layers=args.layers)
    elif args.model_type == 'ugrnn':
        return UGRNN(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          layers=args.layers)
    else:
        raise Exception

model = create_model()
model.double()
params = 0
for p in list(model.parameters()):
    params += p.numel()
print ("Num params: ", params)
print (model)
if args.cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

#Execute neural network on entire input sequence
def execute_sequence(seq, target):
    predicted_list = []
    y_list = []
    model.reset(batch_size=seq.size(0), cuda=args.cuda)

    for i, input_t in enumerate(seq.chunk(seq.size(1), dim=1)):
        input_t = input_t.squeeze(1)
        if activation == None:
            p = model(input_t)
        else:
            p = model(input_t)
            p = activation(p)
        predicted_list.append(p)
        y_list.append(target)

    return predicted_list, y_list

#Train neural network
def train(epoch):
    model.train()
    dset.train()

    total_loss = 0.0
    steps = 0
    n_correct = 0
    n_possible = 0

    #Run batch gradient update
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda().double(), target.cuda().double()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        predicted_list, y_list = execute_sequence(data, target)

        pred = predicted_list[-1]
        y_ = y_list[-1].long()
        prediction = pred.data.max(1, keepdim=True)[1].long()
        n_correct += prediction.eq(y_.data.view_as(prediction)).sum().cpu().numpy()
        n_possible += int(prediction.shape[0])
        loss = F.nll_loss(pred, y_)

        loss.backward()
        optimizer.step()
        steps += 1
        total_loss += loss.cpu().data.numpy()
        optimizer.zero_grad()

    print("Train loss ", total_loss/steps)
    print("Train Acc ", (n_correct/ n_possible))

def validate(epoch):
    dset.val()
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_possible = 0
    steps = 0

    #Run batch inference
    for batch_idx, (data, target) in enumerate(data_loader):

        if args.cuda:
            data, target = data.cuda().double(), target.cuda().double()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            predicted_list, y_list = execute_sequence(data, target)

        pred = predicted_list[-1]
        y_ = y_list[-1].long()
        prediction = pred.data.max(1, keepdim=True)[1].long() #Index of the max log-probability
        n_correct += prediction.eq(y_.data.view_as(prediction)).sum().cpu().numpy()
        n_possible += int(prediction.shape[0])
        loss = F.nll_loss(pred, y_)

        steps += 1
        total_loss += loss.cpu().data.numpy()

    print("Validation Acc ", n_correct/n_possible)
    return total_loss / steps

#General training script logic
def run():
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        print("\n\n**********************************************************")
        tim = time.time()
        train(epoch)
        with torch.no_grad():
            val_loss = validate(epoch)
        print ("Val Loss (epoch", epoch, "): ", val_loss)
        print("Epoch time: ", time.time() - tim)

if __name__ == "__main__":
    run()
