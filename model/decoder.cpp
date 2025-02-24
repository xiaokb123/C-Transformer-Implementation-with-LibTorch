#include <torch/torch.h>
#include "embedding.h"
#include "attention.h"

class DecoderLayerImpl : public torch::nn::Module {
    private:
        torch::nn::MultiheadAttention self_attn_;
        torch::nn::MultiheadAttention enc_attn_;
        torch::nn::Linear feed_forward_;
        torch::nn::Dropout dropout_;

    public:
        DecoderLayerImpl(int64_t d_model, int64_t n_heads, int64_t d_ff, double dropout = 0.1)
            : self_attn_(register_module("self_attn", torch::nn::MultiheadAttention(d_model, n_heads))),
              enc_attn_(register_module("enc_attn", torch::nn::MultiheadAttention(d_model, n_heads))),
              feed_forward_(register_module("feed_forward", torch::nn::Linear(d_model, d_ff))),
              dropout_(register_module("dropout", torch::nn::Dropout(dropout))) {}
        torch::Tensor forward(torch::Tensor x, torch::Tensor enc_output, torch::Tensor mask) {
            // 自注意力
            auto [self_attn_out, _] = self_attn_(x, x, x, mask);
            x = self_attn_out + x;
            x = dropout_->forward(x);

            // 编码器-解码器注意力
            auto [enc_attn_out, __] = enc_attn_(x, enc_output, enc_output, mask);
            x = enc_attn_out + x;
            x = dropout_->forward(x);

            // 前馈网络
            x = feed_forward_->forward(x) + x;
            x = dropout_->forward(x);
            return x;
        }
};
TORCH_MODULE(DecoderLayer);
class DecoderImpl : public torch::nn::Module {
private:
    TokenEmbedding embedding_;
    PositionalEncoding pos_enc_;
    torch::nn::ModuleList layers_;

public:
    DecoderImpl(int64_t vocab_size, int64_t d_model, int64_t n_heads, int64_t n_layers, int64_t d_ff, double dropout)
        : embedding_(register_module("embedding", TokenEmbedding(vocab_size, d_model))),
          pos_enc_(register_module("pos_enc", PositionalEncoding(d_model, dropout))),
          layers_(register_module("layers", torch::nn::ModuleList())) {
        for (int64_t i = 0; i < n_layers; ++i) {
            layers_->push_back(DecoderLayer(d_model, n_heads, d_ff, dropout));
        }
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor enc_output) {
        x = embedding_->forward(x);
        x = pos_enc_->forward(x);
        for (const auto& layer : *layers_) {
            auto decoder_layer = layer->as<DecoderLayer>();
            x = decoder_layer->forward(x, enc_output, torch::Tensor());
        }
        return x;
    }
};

TORCH_MODULE(Decoder);