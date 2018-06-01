import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

'''
Custom UGRNN in PyTorch as seen in:
https://arxiv.org/pdf/1611.09913.pdf
'''
class UGRNNCell(nn.Module):
    def __init__(self, input_size,
                    hidden_size,
                    weight_init=None,
                    reccurent_weight_init=None,
                    drop=None,
                    rec_drop=None):
        super(UGRNNCell, self).__init__()

        print("Initializing UGRNNCell")
        self.hidden_size = hidden_size
        if(weight_init==None):
            self.W_g = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_g = nn.init.xavier_normal_(self.W_g)
            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_c = nn.init.xavier_normal_(self.W_c)
        else:
            self.W_g = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_g = weight_init(self.W_g)
            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_c = weight_init(self.W_c)

        if(reccurent_weight_init == None):
            self.U_g = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_g = nn.init.xavier_normal_(self.U_g)
            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_c = nn.init.xavier_normal_(self.U_c)
        else:
            self.U_g = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_g = recurrent_weight_initializer(self.U_g)
            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_c = recurrent_weight_initializer(self.U_c)

        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

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

        self.states = None

    def reset(self, batch_size=1, cuda=True):
        if cuda:
            self.states = (Variable(torch.zeros(batch_size, self.hidden_size)).cuda().double())
        else:
            self.states = (Variable(torch.zeros(batch_size, self.hidden_size)).double())

    def forward(self, X_t):
        h_t_previous=self.states
        if self.keep_prob:
            X_t = self.dropout(X_t)
        if self.rec_keep_prob:
            h_t_previous = self.rec_dropout(h_t_previous)


        g_t = F.sigmoid(
            torch.mm(X_t, self.W_g) + torch.mm(h_t_previous, self.U_g) + self.b_g #w_f needs to be the previous input shape by the number of hidden neurons
        )

        c_t = F.tanh(
            torch.mm(X_t, self.W_c) + torch.mm(h_t_previous, self.U_c) + self.b_c
        )

        h_t = g_t * h_t_previous + ((g_t - 1) * -1) * c_t


        self.states = h_t
        return h_t

class UGRNN(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 layers=1):
        super(UGRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        self.ugrnns = nn.ModuleList()
        self.ugrnns.append(UGRNNCell(input_size=input_size, hidden_size=hidden_size))
        for i in range(self.layers-1):
            self.ugrnns.append(UGRNNCell(input_size=hidden_size, hidden_size=hidden_size))
        self.fc1 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
        for i in range(len(self.ugrnns)):
            self.ugrnns[i].reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x):
        for i in range(len(self.ugrnns)):
            x = self.ugrnns[i](x)
        o = self.fc1(x)
        return o
