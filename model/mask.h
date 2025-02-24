#pragma once
#include <torch/torch.h>

namespace mask {
    // 创建padding mask，用于处理填充部分
    inline torch::Tensor create_padding_mask(const torch::Tensor& seq) {
        // seq shape: (batch_size, seq_len)
        auto mask = (seq == 0).unsqueeze(1).unsqueeze(2);
        return mask;  // shape: (batch_size, 1, 1, seq_len)
    }

    // 创建因果掩码，用于解码器中防止信息泄露
    inline torch::Tensor create_causal_mask(int64_t size) {
        auto mask = torch::ones({size, size}, torch::kBool);
        return torch::triu(mask, 1).unsqueeze(0);  // shape: (1, size, size)
    }

    // 组合padding mask和因果掩码
    inline torch::Tensor create_combined_mask(const torch::Tensor& seq) {
        int64_t size = seq.size(1);
        auto padding_mask = create_padding_mask(seq);
        auto causal_mask = create_causal_mask(size);
        return torch::max(padding_mask, causal_mask);
    }
} 