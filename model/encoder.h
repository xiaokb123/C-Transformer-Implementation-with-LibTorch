#pragma once
#include <torch/torch.h>
#include "attention.h"
#include "embedding.h"

class EncoderLayerImpl : public torch::nn::Module {
public:
    EncoderLayerImpl(int64_t d_model, int64_t n_heads, int64_t d_ff, double dropout_rate = 0.1);
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask);

private:
    MultiHeadAttention self_attn_;
    torch::nn::Sequential feed_forward_;
    torch::nn::LayerNorm norm1_, norm2_;
};

TORCH_MODULE(EncoderLayer);

class EncoderImpl : public torch::nn::Module {
public:
    EncoderImpl(int64_t vocab_size, int64_t d_model, int64_t n_heads, int64_t n_layers, int64_t d_ff, double dropout);
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = torch::Tensor());

private:
    TokenEmbedding embedding_;
    PositionalEncoding pos_enc_;
    torch::nn::ModuleList layers_;
};

TORCH_MODULE(Encoder); 