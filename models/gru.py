import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

'''
Custom GRU cell and GRU in PyTorch as seen in:
https://arxiv.org/pdf/1406.1078v3.pdf
'''

class GRUCell(nn.Module):
    def __init__(self, input_size,
                    hidden_size,
                    weight_init=None,
                    reccurent_weight_init=None,
                    drop=None,
                    rec_drop=None):
        super(GRUCell, self).__init__()

        print("Initializing GRUCell")
        self.hidden_size = hidden_size

        #Initialize weights for RNN cell
        if(weight_init==None):
            self.W_z = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_z = nn.init.xavier_normal_(self.W_z)
            self.W_r = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_r = nn.init.xavier_normal_(self.W_r)
            self.W_h = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_h = nn.init.xavier_normal_(self.W_h)
        else:
            self.W_z = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_z = weight_init(self.W_z)
            self.W_r = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_r = weight_init(self.W_r)
            self.W_h = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_h = weight_init(self.W_h)

        if(reccurent_weight_init == None):
            self.U_z = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_z = nn.init.orthogonal_(self.U_z)
            self.U_r = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_r = nn.init.orthogonal_(self.U_r)
            self.U_h = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_h = nn.init.orthogonal_(self.U_h)
        else:
            self.U_z = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_z = recurrent_weight_initializer(self.U_z)
            self.U_r = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_r = recurrent_weight_initializer(self.U_r)
            self.U_h = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_h = recurrent_weight_initializer(self.U_h)

        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        #Set up dropout layer if requested
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

        #Initialize recurrent states h_t
        self.recurrent_state = None

    def reset(self, batch_size=1, cuda=True):
        #Reset recurrent states
        if cuda:
            self.recurrent_state = (Variable(torch.randn(batch_size, self.hidden_size)).cuda().double())
        else:
            self.recurrent_state = (Variable(torch.randn(batch_size, self.hidden_size)).double())

    def forward(self, X_t):
        #Define forward calculations for inference time
        h_t_previous = self.recurrent_state

        if self.keep_prob:
            X_t = self.dropout(X_t)
        if self.rec_keep_prob:
            h_t_previous = self.rec_dropout(h_t_previous)
            c_t_previous = self.rec_dropout(c_t_previous)

        z_t = F.sigmoid(
            torch.mm(X_t, self.W_z) + torch.mm(h_t_previous, self.U_z) + self.b_z.expand_as(h_t_previous)
        )

        r_t = F.sigmoid(
            torch.mm(X_t, self.W_r) + torch.mm(h_t_previous, self.U_r) + self.b_r.expand_as(h_t_previous)
        )

        h_t = z_t * h_t_previous + ((z_t - 1) * -1) * F.tanh(
            torch.mm(X_t, self.W_h) + torch.mm((r_t * h_t_previous), self.U_h) + self.b_h.expand_as(h_t_previous)
        )

        self.recurrent_state = h_t
        return h_t

class GRU(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 layers=1):
        super(GRU, self).__init__()
        #Initialize deep RNN neural network

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        #Initialize individual GRU cells
        self.grus = nn.ModuleList()
        self.grus.append(GRUCell(input_size=input_size, hidden_size=hidden_size))

        for index in range(self.layers-1):
            self.grus.append(GRUCell(input_size=hidden_size, hidden_size=hidden_size))

        #Initialize weights for output linear layer
        self.fc1 = nn.Linear(hidden_size, output_size)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
        #Reset recurrent states for all RNN cells defined
        for index in range(len(self.grus)):
            self.grus[index].reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x):
        #Define forward method for deep RNN neural network
        for index in range(len(self.grus)):
            x = self.grus[index](x)
        out = self.fc1(x)
        return out
