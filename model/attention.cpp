// attention.cpp
#include "attention.h"

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t d_model, int64_t n_heads)
    : d_model_(d_model),
      n_heads_(n_heads),
      d_k_(d_model / n_heads),
      query_(register_module("query", torch::nn::Linear(d_model, d_model))),
      key_(register_module("key", torch::nn::Linear(d_model, d_model))),
      value_(register_module("value", torch::nn::Linear(d_model, d_model))),
      out_(register_module("out", torch::nn::Linear(d_model, d_model))) {}

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, 
                                            torch::Tensor mask) {
    auto batch_size = q.size(0);

    // 线性变换
    q = query_->forward(q).view({batch_size, -1, n_heads_, d_k_}).transpose(1, 2);
    k = key_->forward(k).view({batch_size, -1, n_heads_, d_k_}).transpose(1, 2);
    v = value_->forward(v).view({batch_size, -1, n_heads_, d_k_}).transpose(1, 2);

    // 计算注意力
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(d_k_);
    
    if (mask.defined()) {
        // 确保 mask 是布尔类型
        if (mask.scalar_type() != torch::kBool) {
            mask = mask.to(torch::kBool);
        }
        scores = scores.masked_fill(mask, -1e9);
    }

    auto attn = torch::softmax(scores, -1);
    auto context = torch::matmul(attn, v);

    // 重塑并进行最后的线性变换
    context = context.transpose(1, 2).contiguous()
                    .view({batch_size, -1, d_model_});
    
    return out_->forward(context);
}