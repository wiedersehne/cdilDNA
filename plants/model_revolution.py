import torch
import torch.nn as nn


class Revolution_Block(nn.Module):
    def __init__(self, c_in, c_out, hdim, ks, dil, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=hdim, kernel_size=ks, padding='same', dilation=dil, padding_mode='circular', bias=False)
        self.conv2 = nn.Conv1d(in_channels=hdim, out_channels=c_out, kernel_size=ks, padding='same', dilation=dil, padding_mode='circular', bias=False)
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
        return out + res


class Revolution_Conv(nn.Module):
    def __init__(self, dim_in, dim_out, hdim, ks):
        super().__init__()
        layers = []
        for i in range(len(dim_out)):
            this_in = dim_in if i == 0 else dim_out[i - 1]
            this_out = dim_out[i]
            hdim = hdim
            this_dilation = 2 ** i
            this_dropout = 0
            layers += [Revolution_Block(this_in, this_out, hdim, ks, this_dilation, this_dropout)]
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)


class Model4Pretrain(nn.Module):
    def __init__(self, dim, hdim1, hdim2, kernel_size, n_layers):
        super().__init__()

        self.cdilNet = Revolution_Conv(dim, [hdim1]*n_layers, hdim1*2, kernel_size)

        hidden_list = [hdim2]*n_layers
        hidden_list[-1] = dim
        self.decoder = Revolution_Conv(hdim1, hidden_list, hdim2*2, kernel_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.cdilNet(x)
        logits_lm = self.decoder(h)
        return self.sig(logits_lm.permute(0, 2, 1))


class Revolution(nn.Module):
    def __init__(self, dim, hdim1, hdim2, n_targets, kernel_size, n_layers, seq_len, center=200):
        super().__init__()
        self.seq_len = seq_len
        self.center = center

        self.cdilNet = Revolution_Conv(dim, [hdim1]*n_layers, hdim1*2, kernel_size)
        self.cdilNet2 = Revolution_Conv(hdim1, [hdim2] * n_layers, hdim2 * 2, kernel_size)
        self.classifier = nn.Linear(hdim2, n_targets)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.cdilNet(x)
        y_conv = self.cdilNet2(h)
        new_start = int((self.seq_len - self.center) / 2)
        y_conv_new = y_conv[:, :, new_start:new_start+self.center]
        y = self.classifier(torch.mean(y_conv_new, dim=2))
        return self.sig(y)
