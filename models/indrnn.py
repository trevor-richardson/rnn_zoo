import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


'''
Custom INDRNN in PyTorch as seen in:
https://arxiv.org/pdf/1803.04831.pdf
'''

class INDRNNCell(nn.Module):
    def __init__(self, input_size,
                    hidden_size,
                    recurrent_act='relu',
                    weight_init=None,
                    reccurent_weight_init=None,
                    drop=None,
                    rec_drop=None):
        super(INDRNNCell, self).__init__()

        print("Initializing INDRNNCell")
        self.hidden_size = hidden_size
        if(weight_init==None):
            self.W_x = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_x = nn.init.xavier_normal_(self.W_x)
        else:
            self.W_x = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_x = weight_init(self.W_x)

        #Have to initialize to identity matrix
        self.U_h = torch.nn.Parameter(torch.ones(hidden_size))

        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.recurrent_act = recurrent_act

        if(drop==None):
            self.keep_prob = False
        else:
            self.keep_prob = True
            self.dropout = nn.Dropout(drop)
        if(rec_drop == None):
            self.rec_keep_prob = False
        else:
            self.rec_keep_prob = True
            self.rec_dropout = nn.Dropout(rec_drop)


        self.hidden_state = None

    def reset(self, batch_size=1, cuda=True):
        if cuda:
            self.hidden_state = (Variable(torch.randn(batch_size, self.hidden_size)).cuda().double())
        else:
            self.hidden_state = (Variable(torch.randn(batch_size, self.hidden_size)).double())

    def forward(self, X_t):
        h_t_previous = self.hidden_state

        if self.keep_prob:
            X_t = self.dropout(X_t)
        if self.rec_keep_prob:
            h_t_previous = self.rec_dropout(h_t_previous)
            c_t_previous = self.rec_dropout(c_t_previous)

        out = F.relu(
            torch.mm(X_t, self.W_x) + h_t_previous * self.U_h + self.b
        )

        self.hidden_state = out
        return out

class INDRNN(nn.Module):
    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 layers=1,
                 recurrent_act='tanh',
                 use_batchnorm=False):
        super(INDRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.recurrent_act = recurrent_act
        self.use_batchnorm = use_batchnorm


        self.rnns = nn.ModuleList()
        self.rnns.append(INDRNNCell(input_size=input_size, hidden_size=hidden_size, recurrent_act=self.recurrent_act))
        for i in range(self.layers-1):
            self.rnns.append(INDRNNCell(input_size=hidden_size, hidden_size=hidden_size, recurrent_act=self.recurrent_act))
        self.fc1 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
        for i in range(len(self.rnns)):
            self.rnns[i].reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x):

        for i in range(len(self.rnns)):
            x = self.rnns[i](x)
        o = self.fc1(x)

        return o
