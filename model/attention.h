// attention.h
#pragma once
#include <torch/torch.h>
#include <cmath>

class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int64_t d_model, int64_t n_heads);

    torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, 
                         torch::Tensor mask = torch::Tensor());

private:
    int64_t d_model_;
    int64_t n_heads_;
    int64_t d_k_;
    torch::nn::Linear query_{nullptr}, key_{nullptr}, value_{nullptr}, out_{nullptr};
};

TORCH_MODULE(MultiHeadAttention);
