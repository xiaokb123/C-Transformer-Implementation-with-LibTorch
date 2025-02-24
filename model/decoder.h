#ifndef DECODER_H
#define DECODER_H

#include <torch/torch.h>
#include "embedding.h"
#include "attention.h"

class DecoderLayerImpl : public torch::nn::Module {
public:
    DecoderLayerImpl(int64_t d_model, int64_t n_heads, int64_t d_ff, double dropout = 0.1)
        : self_attn_(register_module("self_attn", MultiHeadAttention(d_model, n_heads))),
          enc_attn_(register_module("enc_attn", MultiHeadAttention(d_model, n_heads))),
          feed_forward_(register_module("feed_forward", 
              torch::nn::Sequential(
                  torch::nn::Linear(d_model, d_ff),
                  torch::nn::ReLU(),
                  torch::nn::Linear(d_ff, d_model)
              ))),
          norm1_(register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})))),
          norm2_(register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})))),
          norm3_(register_module("norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})))),
          dropout_(register_module("dropout", torch::nn::Dropout(dropout))) {}

    torch::Tensor forward(torch::Tensor x, torch::Tensor enc_output, 
                         torch::Tensor self_mask, torch::Tensor enc_mask) {
        auto attn1 = self_attn_->forward(x, x, x, self_mask);
        x = norm1_(x + dropout_->forward(attn1));

        auto attn2 = enc_attn_->forward(x, enc_output, enc_output, enc_mask);
        x = norm2_(x + dropout_->forward(attn2));

        auto ff = feed_forward_->forward(x);
        x = norm3_(x + dropout_->forward(ff));

        return x;
    }

private:
    MultiHeadAttention self_attn_;
    MultiHeadAttention enc_attn_;
    torch::nn::Sequential feed_forward_;
    torch::nn::LayerNorm norm1_, norm2_, norm3_;
    torch::nn::Dropout dropout_;
};

TORCH_MODULE(DecoderLayer);

class DecoderImpl : public torch::nn::Module {
public:
    DecoderImpl(int64_t vocab_size, int64_t d_model, int64_t n_heads, 
                int64_t n_layers, int64_t d_ff, double dropout)
        : embedding_(register_module("embedding", TokenEmbedding(vocab_size, d_model))),
          pos_enc_(register_module("pos_enc", PositionalEncoding(d_model, dropout))) {
        layers_ = register_module("layers", torch::nn::ModuleList());
        for (int i = 0; i < n_layers; ++i) {
            layers_->push_back(DecoderLayer(d_model, n_heads, d_ff, dropout));
        }
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor enc_output,
                         torch::Tensor self_mask = torch::Tensor(),
                         torch::Tensor enc_mask = torch::Tensor()) {
        x = embedding_->forward(x);
        x = pos_enc_->forward(x);

        for (auto& layer : *layers_) {
            x = layer->as<DecoderLayerImpl>()->forward(x, enc_output, self_mask, enc_mask);
        }

        return x;
    }

private:
    TokenEmbedding embedding_;
    PositionalEncoding pos_enc_;
    torch::nn::ModuleList layers_;
};

TORCH_MODULE(Decoder);

#endif 