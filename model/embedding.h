// embedding.h
#pragma once
#include <torch/torch.h>
#include <cmath>

// TokenEmbedding模块声明
class TokenEmbeddingImpl : public torch::nn::Module {
public:
    TokenEmbeddingImpl(int64_t vocab_size, int64_t d_model);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Embedding embedding_{nullptr};
};

TORCH_MODULE(TokenEmbedding);

// PositionalEncoding模块声明
class PositionalEncodingImpl : public torch::nn::Module {
public:
    PositionalEncodingImpl(int64_t d_model, double dropout = 0.1, int64_t max_len = 5000);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Dropout dropout_{nullptr};
    torch::Tensor pe;
};

TORCH_MODULE(PositionalEncoding);