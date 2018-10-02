import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

'''
Custom IntersectionRNN built in PyTorch as seen in:
https://arxiv.org/pdf/1611.09913.pdf
'''

class IntersectionRNNCell(nn.Module):
    def __init__(self, input_size,
                    hidden_size,
                    weight_init=None,
                    reccurent_weight_init=None,
                    drop=None,
                    rec_drop=None):
        super(IntersectionRNNCell, self).__init__()

        print("Initializing IntersectionRNNCell")

        #Initialize weights for RNN cell
        self.hidden_size = hidden_size
        if(weight_init==None):
            self.W_yin = nn.Parameter(torch.zeros(input_size, input_size))
            self.W_yin = nn.init.xavier_normal_(self.W_yin)
            self.W_hin = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_hin = nn.init.xavier_normal_(self.W_hin)
            self.W_gy = nn.Parameter(torch.zeros(input_size, input_size))
            self.W_gy = nn.init.xavier_normal_(self.W_gy)
            self.W_gh = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_gh = nn.init.xavier_normal_(self.W_gh)
        else:
            self.W_yin = nn.Parameter(torch.zeros(input_size, input_size))
            self.W_yin = weight_init(self.W_yin)
            self.W_hin = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_hin = weight_init(self.W_hin)
            self.W_gy = nn.Parameter(torch.zeros(input_size, input_size))
            self.W_gy = weight_init(self.W_gy)
            self.W_gh = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_gh = weight_init(self.W_gh)

        if(reccurent_weight_init == None):
            self.U_yin = nn.Parameter(torch.zeros(hidden_size, input_size))
            self.U_yin = nn.init.orthogonal_(self.U_yin)
            self.U_hin = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_hin = nn.init.orthogonal_(self.U_hin)
            self.U_gy = nn.Parameter(torch.zeros(hidden_size, input_size))
            self.U_gy = nn.init.orthogonal_(self.U_gy)
            self.U_gh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_gh = nn.init.orthogonal_(self.U_gh)
        else:
            self.U_yin = nn.Parameter(torch.zeros(input_size, input_size))
            self.U_yin = recurrent_weight_initializer(self.U_yin)
            self.U_hin = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_hin = recurrent_weight_initializer(self.U_hin)
            self.U_gy = nn.Parameter(torch.zeros(input_size, input_size))
            self.U_gy = recurrent_weight_initializer(self.U_gy)
            self.U_gh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_gh = recurrent_weight_initializer(self.U_gh)

        self.b_yin = nn.Parameter(torch.zeros(input_size))
        self.b_hin = nn.Parameter(torch.zeros(hidden_size))
        self.b_gy = nn.Parameter(torch.zeros(input_size))
        self.b_gh = nn.Parameter(torch.zeros(hidden_size))

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

        #Initialize recurrent states h_t
        self.states = None

    def reset(self, batch_size=1, cuda=True):
        #Reset recurrent states
        if cuda:
            self.states = (Variable(torch.zeros(batch_size, self.hidden_size)).cuda().double())
        else:
            self.states = (Variable(torch.zeros(batch_size, self.hidden_size)).double())

    def forward(self, X_t):
        #Define forward calculations for inference time
        h_t_previous = self.states

        if self.keep_prob:
            X_t = self.dropout(X_t)
        if self.rec_keep_prob:
            h_t_previous = self.rec_dropout(h_t_previous)

        y_in = torch.tanh(
            torch.mm(X_t, self.W_yin) + torch.mm(h_t_previous, self.U_yin) + self.b_yin #w_f needs to be the previous input shape by the number of hidden neurons
        )

        h_in = torch.tanh(
            torch.mm(X_t, self.W_hin) + torch.mm(h_t_previous, self.U_hin) + self.b_hin
        )

        g_y = torch.sigmoid(
            torch.mm(X_t, self.W_gy) + torch.mm(h_t_previous, self.U_gy) + self.b_gy
        )

        g_h = torch.sigmoid(
            torch.mm(X_t, self.W_gh) + torch.mm(h_t_previous, self.U_gh) + self.b_gh
        )

        y_t = g_y * X_t + ((g_y - 1) * -1) * y_in

        h_t = g_h * h_t_previous + ((g_h - 1) *-1) * h_in

        self.states = h_t
        return y_t

class IntersectionRNN(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 layers=1,
                 drop=None,
                 rec_drop=None):
        super(IntersectionRNN, self).__init__()
        #Initialize deep RNN neural network

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        #Initialize individual RNN+ cells
        self.int_rnns = nn.ModuleList()
        self.int_rnns.append(IntersectionRNNCell(input_size=input_size, hidden_size=hidden_size, drop=drop, rec_drop=rec_drop))
        for index in range(self.layers-1):
            self.int_rnns.append(IntersectionRNNCell(input_size=input_size, hidden_size=hidden_size, drop=drop, rec_drop=rec_drop))

        #Initialize weights for output linear layer
        self.fc1 = nn.Linear(input_size, output_size)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
        #Reset recurrent states for all RNN cells defined
        for index in range(len(self.int_rnns)):
            self.int_rnns[index].reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x):
        #Define forward method for deep RNN neural network
        for index in range(len(self.int_rnns)):
            x = self.int_rnns[index](x)
        out = self.fc1(x)
        return out
