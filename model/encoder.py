# Encoder 模块
# models/encoder.py
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import TokenEmbedding, PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.self_attn(x, x, x, mask) + x
        x = self.feed_forward(x) + x
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, None)
        return x