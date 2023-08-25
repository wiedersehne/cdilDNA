import math
import torch
import torch.nn as nn
from linformer import Linformer
from performer_pytorch import Performer
from nystrom_attention import Nystromformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


#  Refer to https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]  # [seq_len, batch_size, dim]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class XFormer(nn.Module):
    def __init__(self, model, use_pos, input_size, dim, depth, heads, seq_len):
        super(XFormer, self).__init__()

        self.model = model
        self.use_pos = use_pos
        if use_pos:
            self.linear = nn.Linear(input_size, dim)
            self.pos_enc = PositionalEncoding(dim, seq_len)

        if model == 'transformer':
            encoder_layers = TransformerEncoderLayer(dim, heads, dim)
            self.former = TransformerEncoder(encoder_layers, depth)
        elif model == 'performer':
            self.former = Performer(dim=dim, depth=depth, heads=heads, dim_head=dim)
        elif model == 'linformer':
            self.former = Linformer(dim=dim, seq_len=seq_len, depth=depth, heads=heads, k=dim)
        elif model == 'nystromformer':
            self.former = Nystromformer(dim=dim, depth=depth, heads=heads)

    def forward(self, x):
        if self.use_pos:
            x = self.linear(x)
            x = self.pos_enc(x)  # out: num, length, dim

        if self.model == 'transformer':
            x = x.permute(1, 0, 2)
            x = self.former(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.former(x)

        return x


class FormerClassifier(nn.Module):
    def __init__(self, name, layers, heads, dim_in, dim_out, clf_dim, output_size, max_len):
        super(FormerClassifier, self).__init__()

        self.encoder = XFormer(name, False, dim_in, dim_out, depth=layers, heads=heads, seq_len=max_len)
        self.Net2 = XFormer(name, False, dim_out, clf_dim, depth=layers, heads=heads, seq_len=max_len)
        self.classifier = nn.Linear(clf_dim, output_size)
        # self.sig = nn.Sigmoid()

    def freeze_cdilNet(self):
        for param in self.cdilNet.parameters():
            param.requires_grad = False

    def forward(self, x1, x2, idx_linear):
        x1, x2 = x1.float(), x2.float()
        idx_linear = idx_linear.to(torch.int64)
        x1, x2 = x1.permute(0, 2, 1), x2.permute(0, 2, 1)
        # x1, x2 = F.pad(x1, (0, 0, 0, 24, 0, 0)), F.pad(x2, (0, 0, 0, 24, 0, 0))
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        y_vcf = y1 - y2
        y_vcf = self.Net2(y_vcf)
        y_class = torch.mean(y_vcf, dim=1)
        y = self.classifier(y_class)
        idx_linear = idx_linear.unsqueeze(0).t()
        y = torch.gather(y, 1, idx_linear)
        return y


class FormerClassifier_plant(nn.Module):
    def __init__(self, name, depth1, depth2, heads, dim_in, dim_out, output_size, max_len, center=200):
        super().__init__()
        self.seq_len = max_len
        self.center = center

        self.encoder = XFormer(name, True, dim_in, dim_out, depth=depth1, heads=heads, seq_len=max_len)
        self.Net2 = XFormer(name, False, dim_out, dim_out, depth=depth2, heads=heads, seq_len=max_len)
        self.linear = nn.Linear(dim_out, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, seq):
        en = self.encoder(seq)
        de = self.Net2(en)
        new_start = int((self.seq_len - self.center) / 2)
        y_center = de[:, new_start:new_start + self.center, :]
        y = self.linear(torch.mean(y_center, dim=1))
        return self.sig(y)
