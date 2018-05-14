import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class dnn_model(nn.Module):

    def __init__(self, vocab_size = 3500, fc1_num = 128, fc2_num = 32):
        super(dnn_model, self).__init__()
        self.fc1_l =  nn.Linear(vocab_size, fc1_num)
        self.fc1_r = nn.Linear(vocab_size, fc1_num)
        self.fc2 =  nn.Linear(fc1_num*2, fc2_num)
        self.fc3 =  nn.Linear(fc2_num, 1)

    def forward(self, inputs):
        x_l = self.fc1_l(inputs[0])
        x_r = self.fc1_r(inputs[1])
        x_l = F.sigmoid(x_l)
        x_r = F.sigmoid(x_r)
        x = torch.cat([x_l, x_r], 1)        
        x = F.dropout(x, p = 0.25)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.dropout(x, p = 0.25)
        x = self.fc3(x)
        x = F.sigmoid(x)

        return x
