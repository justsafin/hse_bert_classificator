import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, batch_first=True, dropout=0):
        super(FFN, self).__init__()
        self.gru = nn.GRU(d_model, d_model * 2, 1, bidirectional=bidirectional, batch_first=batch_first)
        if bidirectional:
            self.linear = nn.Linear(d_model * 2 * 2, d_model)
        else:
            self.linear = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, batch_first=True, dropout=0):
        super(SelfAttention, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first)

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, batch_first=batch_first, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, batch_first=True, dropout=0):
        super(CrossAttention, self).__init__()

        self.norm1_q = nn.LayerNorm(d_model)
        self.norm1_kv = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first)

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, batch_first=batch_first, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, kv):
        qt = self.norm1_q(x)
        kvt = self.norm1_kv(kv)
        xt, _ = self.attention(qt, kvt, kvt)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x


class FusionBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, batch_first=True, dropout=0):
        super(FusionBlock, self).__init__()
        self.cross_attn = CrossAttention(d_model, n_heads, bidirectional=bidirectional, batch_first=batch_first,
                                         dropout=dropout)
        self.self_attn = SelfAttention(d_model, n_heads, bidirectional=bidirectional, batch_first=batch_first,
                                       dropout=dropout)

    def forward(self, x, last_layer):
        x = x + self.cross_attn(x, last_layer)
        x = x + self.self_attn(x)
        return x


class SqueezeBlock(nn.Module):
    def __init__(self, d_model):
        super(SqueezeBlock, self).__init__()
        self.seq = nn.Sequential(nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
                                 nn.LayerNorm(256),
                                 nn.PReLU(),
                                 nn.Conv1d(d_model // 2, 1, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.seq(x)
        x = torch.squeeze(x)
        return x
