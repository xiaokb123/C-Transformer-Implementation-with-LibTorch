#include "embedding.h"

TokenEmbeddingImpl::TokenEmbeddingImpl(int64_t vocab_size, int64_t d_model) {
    embedding_ = register_module("embedding", 
        torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, d_model)));
}

torch::Tensor TokenEmbeddingImpl::forward(torch::Tensor x) {
    return embedding_->forward(x);
}

PositionalEncodingImpl::PositionalEncodingImpl(int64_t d_model, double dropout, int64_t max_len) {
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    
    pe = torch::zeros({max_len, d_model});
    auto position = torch::arange(0, max_len, torch::kFloat32).unsqueeze(1);
    auto div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat32) *
                             (-std::log(10000.0) / d_model));

    pe.slice(1, 0, d_model, 2) = torch::sin(position * div_term);
    pe.slice(1, 1, d_model, 2) = torch::cos(position * div_term);
    pe = pe.unsqueeze(0);

    register_buffer("pe", pe);
}

torch::Tensor PositionalEncodingImpl::forward(torch::Tensor x) {
    auto pe_subset = pe.slice(1, 0, x.size(1));
    x = x + pe_subset;
    return dropout_->forward(x);
}