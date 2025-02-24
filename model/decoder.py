# Decoder 模块
# models/decoder.py
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import TokenEmbedding, PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, mask):
        x = self.self_attn(x, x, x, mask) + x
        x = self.enc_attn(x, enc_output, enc_output, mask) + x
        x = self.feed_forward(x) + x
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_output):
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_output, None)
        return x