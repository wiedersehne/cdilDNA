import torch
import torch.nn as nn


class CDIL_Block(nn.Module):
    def __init__(self, c_in, c_out, hdim, ks, dil, dropout):
        super(CDIL_Block).__init__()
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
        return out + res


class CDIL_Layer(nn.Module):
    def __init__(self, dim_in, dim_out, hdim, ks, dropout):
        super(CDIL_Layer).__init__()
        layers = []
        for i in range(len(dim_out)):
            current_input = dim_in if i == 0 else dim_out[i - 1]
            current_output = dim_out[i]
            hdim = hdim
            current_dilation = 2 ** i
            current_dropout = dropout
            layers += [CDIL_Block(current_input, current_output, hdim, ks, current_dilation, current_dropout)]
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


class Classifier_human(nn.Module):
    def __init__(self, dim_in, dim_out, clf_dim, layers, ks, output_size, max_len, dropout):
        super(Classifier_human, self).__init__()
        self.encoder = CDIL_Layer(dim_in, [dim_out]*layers, dim_out*2, ks, dropout)
        self.decoder = CDIL_Layer(dim_out, [clf_dim]*layers, clf_dim*2, ks, dropout)
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
        y = self.decoder(y)
        y = torch.mean(y, dim=2)
        y = self.classifier(y)
        idx_linear = idx_linear.unsqueeze(0).t().type(torch.int64)
        y = torch.gather(y, 1, idx_linear)
        return y


class Classifier_plant(nn.Module):
    def __init__(self, dim_in, dim_out, clf_dim, layers, ks, output_size, max_len, dropout, center=200):
        super(Classifier_plant, self).__init__()
        self.seq_len = max_len
        self.center = center

        self.encoder = CDIL_Layer(dim_in, [dim_out]*layers, dim_out*2, ks, dropout)
        self.decoder = CDIL_Layer(dim_out, [clf_dim]*layers, clf_dim*2, ks, dropout)
        self.classifier = ClassifierHead(clf_dim, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.encoder(x)
        y = self.decoder(y)
        new_start = int((self.seq_len - self.center) / 2)
        y_conv_new = y[:, :, new_start:new_start + self.center]
        y = self.classifier(torch.mean(y_conv_new, dim=2))
        return y


class Model4Pretrain(nn.Module):
    def __init__(self, dim_in, dim_out, clf_dim, layers, ks, dropout):
        super(Model4Pretrain, self).__init__()

        self.encoder = CDIL_Layer(dim_in, [dim_out]*layers, dim_out*2, ks, dropout)
        self.decoder = CDIL_Layer(dim_out, [clf_dim]*layers, clf_dim*2, ks, dropout)
        self.classifier = ClassifierHead(clf_dim, dim_in)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.encoder(x)
        y = self.decoder(y)
        y = self.classifier(y.permute(0, 2, 1)).permute(0, 2, 1)
        return self.sig(y)
