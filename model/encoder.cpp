#include <torch/torch.h>
#include <cmath>
#include "attention.h"
#include "embedding.h"
#include "encoder.h"

EncoderLayerImpl::EncoderLayerImpl(int64_t d_model, int64_t n_heads, int64_t d_ff, double dropout_rate)
    : self_attn_(register_module("self_attn", MultiHeadAttention(d_model, n_heads))),
      feed_forward_(register_module("feed_forward",
          torch::nn::Sequential(
              torch::nn::Linear(d_model, d_ff),
              torch::nn::ReLU(),
              torch::nn::Linear(d_ff, d_model),
              torch::nn::Dropout(dropout_rate)
          ))),
      norm1_(register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})))),
      norm2_(register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})))) {}

torch::Tensor EncoderLayerImpl::forward(torch::Tensor x, torch::Tensor mask) {
    // 自注意力子层
    torch::Tensor attn_out = self_attn_->forward(x, x, x, mask);
    x = norm1_(x + attn_out);

    // 前馈子层
    torch::Tensor ff_out = feed_forward_->forward(x);
    return norm2_(x + ff_out);
}

EncoderImpl::EncoderImpl(int64_t vocab_size, int64_t d_model, int64_t n_heads, int64_t n_layers, int64_t d_ff, double dropout)
    : embedding_(register_module("embedding", TokenEmbedding(vocab_size, d_model))),
      pos_enc_(register_module("pos_enc", PositionalEncoding(d_model, dropout))) {
    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int i = 0; i < n_layers; ++i) {
        layers_->push_back(EncoderLayer(d_model, n_heads, d_ff, dropout));
    }
}

torch::Tensor EncoderImpl::forward(torch::Tensor x, torch::Tensor mask) {
    x = embedding_->forward(x);
    x = pos_enc_->forward(x);
    for (auto& layer : *layers_) {
        x = layer->as<EncoderLayerImpl>()->forward(x, mask);
    }
    return x;
}