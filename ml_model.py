import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class symcnn_model(nn.Module):

    def __init__(self, vocab_size = 3500, embedding_dim = 128, context_size = 20, conv_num_kernel1 = 32, conv_num_kernel2 = 64, fc1_num = 128, fc2_num = 32, kernel_size1 = 5, kernel_size2 = 3, pool1_size = 2):
        super(symcnn_model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #self.left_conv  = nn.Conv1d(in_channels = embedding_dim, out_channels = conv_num_kernel, kernel_size = 5)
        self.conv1 = nn.Conv1d(in_channels = embedding_dim, out_channels = conv_num_kernel1, kernel_size = kernel_size1)
        self.conv2 = nn.Conv1d(in_channels = conv_num_kernel1, out_channels = conv_num_kernel2, kernel_size = kernel_size2)
        self.fc1 =  nn.Linear(2*conv_num_kernel2, fc1_num)
        self.fc2 =  nn.Linear(fc1_num, fc2_num)
        self.fc3 =  nn.Linear(fc2_num, 1)
        self.pool1 = nn.MaxPool1d(kernel_size = pool1_size)

    def cnn(self, x):
        x = torch.transpose(self.embeddings(x),1,2)
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.pool1(x)   
        x = F.dropout(x, p=0.5)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = (F.avg_pool1d(x, kernel_size = x.size()[2])).squeeze()
        x = F.dropout(x, p=0.5)
        return x

    def forward(self, inputs):
        left_x = self.cnn(inputs[0]) 
        right_x = self.cnn(inputs[1])       
        x = torch.cat([left_x, right_x], 1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, p = 0.5)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.dropout(x, p = 0.5)
        x = self.fc3(x)
        x = F.sigmoid(x)

        return x
