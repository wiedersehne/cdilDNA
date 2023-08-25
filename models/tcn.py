import torch
import torch.nn as nn


# TCN
class tcn(nn.Module):
    def __init__(self, tcn_size):
        super(tcn, self).__init__()
        self.tcn_size = tcn_size

    def forward(self, x):
        x_new = x[:, :, :-self.tcn_size]
        return x_new.contiguous()


class TCN_Block(nn.Module):
    def __init__(self, c_in, c_out, hdim, ks, dil, dropout, pad):
        super(TCN_Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=hdim, kernel_size=ks, padding=pad, dilation=dil, padding_mode='zeros', bias=False)
        self.cut1 = tcn(pad)
        self.conv2 = nn.Conv1d(in_channels=hdim, out_channels=c_out, kernel_size=ks, padding=pad, dilation=dil, padding_mode='zeros', bias=False)
        self.cut2 = tcn(pad)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hdim)
        self.batch_norm2 = nn.BatchNorm1d(c_out)
        self.nonlinear1 = nn.ReLU()
        self.nonlinear2 = nn.ReLU()
        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.cut1(out)
        out = self.dropout1(out)
        out = self.batch_norm1(out)
        out = self.nonlinear1(out)

        out = self.conv2(out)
        out = self.cut2(out)
        out = self.dropout2(out)
        out = self.batch_norm2(out)
        out = self.nonlinear2(out)

        res = x if self.res is None else self.res(x)
        return out + res


class TCN_Layer(nn.Module):
    def __init__(self, dim_in, dim_out, hdim, ks, dropout):
        super(TCN_Layer, self).__init__()
        layers = []
        for i in range(len(dim_out)):
            current_input = dim_in if i == 0 else dim_out[i - 1]
            current_output = dim_out[i]
            hdim = hdim
            current_dilation = 2 ** i
            current_padding = current_dilation * (ks - 1)
            current_dropout = dropout
            layers += [TCN_Block(current_input, current_output, hdim, ks, current_dilation, current_dropout, current_padding)]
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


class ClassifierTCN_plant(nn.Module):
    def __init__(self, dim_in, dim_out, clf_dim, layers, ks, output_size, max_len, dropout, center=200):
        super(ClassifierTCN_plant, self).__init__()
        self.seq_len = max_len
        self.center = center

        self.encoder = TCN_Layer(dim_in, [dim_out]*layers, dim_out*2, ks, dropout)
        self.decoder = TCN_Layer(dim_out, [clf_dim]*layers, clf_dim*2, ks, dropout)
        self.classifier = ClassifierHead(clf_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.encoder(x)
        y = self.decoder(y)
        new_start = int((self.seq_len - self.center) / 2)
        y_conv_new = y[:, :, new_start:new_start + self.center]
        y = self.classifier(torch.mean(y_conv_new, dim=2))
        return self.sig(y)
