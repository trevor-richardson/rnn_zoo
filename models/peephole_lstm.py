import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

'''
Custom Built Peephole LSTM in PyTorch as seen in:
ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf
'''

class PeepholeCell(nn.Module):
    def __init__(self, input_size,
                    hidden_size,
                    weight_init=None,
                    reccurent_weight_init=None,
                    drop=None,
                    rec_drop=None):
        super(PeepholeCell, self).__init__()

        print("Initializing PeepholeCell")
        self.hidden_size = hidden_size

        #Initialize weights for RNN cell
        if(weight_init==None):
            self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_f = nn.init.xavier_normal_(self.W_f)
            self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_i = nn.init.xavier_normal_(self.W_i)
            self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_o = nn.init.xavier_normal_(self.W_o)
            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_c = nn.init.xavier_normal_(self.W_c)
        else:
            self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_f = weight_init(self.W_f)
            self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_i = weight_init(self.W_i)
            self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_o = weight_init(self.W_o)
            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_c = weight_init(self.W_c)

        if(reccurent_weight_init == None):
            self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_f = nn.init.orthogonal_(self.U_f)
            self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_i = nn.init.orthogonal_(self.U_i)
            self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_o = nn.init.orthogonal_(self.U_o)
            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_c = nn.init.orthogonal_(self.U_c)
        else:
            self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_f = recurrent_weight_initializer(self.U_f)
            self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_i = recurrent_weight_initializer(self.U_i)
            self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_o = recurrent_weight_initializer(self.U_o)
            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_c = recurrent_weight_initializer(self.U_c)

        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        #Set up dropout layer if requested
        if(drop==0):
            self.keep_prob = False
        else:
            self.keep_prob = True
            self.dropout = nn.Dropout(drop)
        if(rec_drop == 0):
            self.rec_keep_prob = False
        else:
            self.rec_keep_prob = True
            self.rec_dropout = nn.Dropout(rec_drop)

        #Initialize recurrent states h_t and c_t
        self.states = None

    def reset(self, batch_size=1, cuda=True):
        #Reset recurrent states
        if cuda:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)).cuda().double(), Variable(torch.randn(batch_size, self.hidden_size)).cuda().double())
        else:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)).double(), Variable(torch.randn(batch_size, self.hidden_size)).double())

    def forward(self, X_t):
        #Define forward calculations for inference time
        h_t_previous, c_t_previous = self.states


        if self.keep_prob:
            X_t = self.dropout(X_t)
        if self.rec_keep_prob:
            h_t_previous = self.rec_dropout(h_t_previous)
            c_t_previous = self.rec_dropout(c_t_previous)

        f_t = torch.sigmoid(
            torch.mm(X_t, self.W_f) + torch.mm(c_t_previous, self.U_f) + self.b_f
        )


        i_t = torch.sigmoid(
            torch.mm(X_t, self.W_i) + torch.mm(c_t_previous, self.U_i) + self.b_i
        )


        o_t = torch.sigmoid(
            torch.mm(X_t, self.W_o) + torch.mm(c_t_previous, self.U_o) + self.b_o
        )


        c_hat_t = torch.tanh(
            torch.mm(X_t, self.W_c) + torch.mm(c_t_previous, self.U_c) + self.b_c
        )

        c_t = (f_t * c_t_previous) + (i_t * c_hat_t)

        h_t = o_t * torch.tanh(c_t)

        self.states = (h_t, c_t)
        return h_t

class Peephole(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 layers=1,
                 drop=None,
                 rec_drop=None):
        super(Peephole, self).__init__()
        #Initialize deep RNN neural network

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        #Initialize individual Peephole cells
        self.lstms = nn.ModuleList()
        self.lstms.append(PeepholeCell(input_size=input_size, hidden_size=hidden_size, drop=drop, rec_drop=rec_drop))

        for index in range(self.layers-1):
            self.lstms.append(PeepholeCell(input_size=hidden_size, hidden_size=hidden_size, drop=drop, rec_drop=rec_drop))

        #Initialize weights for output linear layer
        self.fc1 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
        #Reset recurrent states for all RNN cells defined
        for index in range(len(self.lstms)):
            self.lstms[index].reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x):
        #Define forward method for deep RNN neural network
        for index in range(len(self.lstms)):
            x = self.lstms[index](x)
        out = self.fc1(x)
        return out
