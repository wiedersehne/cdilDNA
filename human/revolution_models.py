import math
import json
from typing import NamedTuple
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import BatchNorm1d


class Config(NamedTuple):
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    n_class: int = 919 # Number of classes
    promote: str = "True" #if use special tokens at the beginning
    hdim: int = 128

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class RevolutionBlock(nn.Module):
    def __init__(self, c_in, c_out, hdim, ks, dil, dropout):
        super(RevolutionBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=hdim, kernel_size=ks, padding='same', dilation=dil, padding_mode='circular', bias = False)
        self.conv2 = nn.Conv1d(in_channels=hdim, out_channels=c_out, kernel_size=ks, padding='same', dilation=dil, padding_mode='circular', bias = False)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hdim)
        self.batch_norm2 = nn.BatchNorm1d(c_out)
        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.batch_norm1(out)
        out = self.nonlinear(out)
        
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.batch_norm2(out)
        out = self.nonlinear(out)
        res = x if self.res is None else self.res(x)
        return self.nonlinear(out) + res
    
# class RevolutionBlock2(nn.Module):
#     def __init__(self, c_in, c_out, hdim, ks, dil, dropout):

#         super().__init__()
#         self.conv = nn.Conv1d(in_channels=c_in, out_channels=hdim, kernel_size=ks, padding='same', dilation=dil, padding_mode='circular')

#         self.layer_norm1 = nn.LayerNorm(hdim)
#         self.nonlinear1 = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm2 = nn.LayerNorm(hdim)
#         self.conv21 = nn.Conv1d(in_channels=hdim, out_channels=hdim*2, kernel_size=1)
#         self.nonlinear2 = nn.ReLU()
#         self.conv22 = nn.Conv1d(in_channels=hdim*2, out_channels=c_out, kernel_size=1)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.layer_norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
#         out = self.conv(x)
#         out = self.dropout(self.nonlinear1(out))
#         x2 = out + x
#         x2 = self.layer_norm2(x2.permute(0, 2, 1)).permute(0, 2, 1)
#         out2 = self.dropout2(self.conv22(self.nonlinear2(self.conv21(x2))))
#         return out2 + x2


class RevolutionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, hdim, ks, dropout):
        super(RevolutionLayer, self).__init__()
        layers = []
        for i in range(len(dim_out)):
            current_input = dim_in if i == 0 else dim_out[i - 1]
            current_output = dim_out[i]
            hdim = hdim
            current_dilation = 2 ** i
            current_dropout = dropout
            layers += [RevolutionBlock(current_input, current_output, hdim, ks, current_dilation, current_dropout)]
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)


class ClassifierHead(nn.Module):
    def __init__(self, dim_hidden, out):
        super(ClassifierHead, self).__init__()
        self.linear = nn.Linear(dim_hidden, out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.linear(x)
        return y


class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out, clf_dim, layers, ks, output_size, max_len, dropout):
        super(Classifier, self).__init__()
        self.encoder = RevolutionLayer(dim_in, [dim_out]*layers, dim_out*2, ks, dropout)
        self.revoClf = RevolutionLayer(dim_out, [clf_dim]*layers, clf_dim*2, ks, dropout)
        self.classifier = ClassifierHead(clf_dim, output_size)
        # self.freeze_cdilNet()dim_in: 5

    def freeze_cdilNet(self):
        for param in self.cdilNet.parameters():
            param.requires_grad = False

    def forward(self, x1, x2, idx_linear):
        # print(x1.shape, x2.shape)
        x1, x2 = x1.float(), x2.float()
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        y = y1 - y2
        y = self.revoClf(y)
        y = torch.mean(y, dim=2)
        y = self.classifier(y)
        idx_linear = idx_linear.unsqueeze(0).t().type(torch.int64)
        y = torch.gather(y, 1, idx_linear)
        return y
