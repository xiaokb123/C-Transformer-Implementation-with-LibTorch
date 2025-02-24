#pragma once
#include <torch/torch.h>
#include "encoder.h"
#include "decoder.h"
#include "mask.h"
#include <fstream>

class TransformerImpl : public torch::nn::Module {
public:
    TransformerImpl(int64_t vocab_size, int64_t d_model, int64_t n_heads, 
                   int64_t n_layers, int64_t d_ff, double dropout = 0.1)
        : encoder_(register_module("encoder", Encoder(vocab_size, d_model, n_heads, n_layers, d_ff, dropout))),
          decoder_(register_module("decoder", Decoder(vocab_size, d_model, n_heads, n_layers, d_ff, dropout))),
          projection_(register_module("projection", torch::nn::Linear(d_model, vocab_size))) {}

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt) {
        // 创建掩码
        auto src_mask = mask::create_padding_mask(src);
        auto tgt_mask = mask::create_combined_mask(tgt);
        
        // 前向传播
        torch::Tensor enc_output = encoder_->forward(src, src_mask);
        torch::Tensor dec_output = decoder_->forward(tgt, enc_output, tgt_mask, src_mask);
        return projection_->forward(dec_output);
    }

    // 计算损失
    torch::Tensor compute_loss(torch::Tensor logits, torch::Tensor targets, torch::Tensor pad_mask) {
        auto loss_fn = torch::nn::CrossEntropyLoss();
        auto loss = loss_fn->forward(logits.reshape({-1, logits.size(-1)}), 
                                   targets.reshape({-1}));
        return loss;
    }

    // 训练一个批次
    std::tuple<torch::Tensor, float> train_step(
        torch::Tensor src, torch::Tensor tgt, torch::Tensor tgt_y,
        torch::optim::Optimizer& optimizer) {
        
        optimizer.zero_grad();
        
        // 前向传播
        auto logits = this->forward(src, tgt);
        
        // 计算损失
        auto pad_mask = (tgt_y != 0);
        auto loss = compute_loss(logits, tgt_y, pad_mask);
        
        // 反向传播
        loss.backward();
        
        // 梯度裁剪
        torch::nn::utils::clip_grad_norm_(this->parameters(), 1.0);
        
        // 优化器步进
        optimizer.step();
        
        return std::make_tuple(logits, loss.item<float>());
    }

    // 保存模型
    void save(const std::string& path) {
        torch::save(encoder_, path + ".encoder");
        torch::save(decoder_, path + ".decoder");
        torch::save(projection_, path + ".projection");
    }

    // 加载模型
    void load(const std::string& path) {
        torch::load(encoder_, path + ".encoder");
        torch::load(decoder_, path + ".decoder");
        torch::load(projection_, path + ".projection");
    }

    // 验证步骤
    std::tuple<torch::Tensor, float> validate_step(
        torch::Tensor src, torch::Tensor tgt, torch::Tensor tgt_y) {
        torch::NoGradGuard no_grad;
        
        auto logits = this->forward(src, tgt);
        auto pad_mask = (tgt_y != 0);
        auto loss = compute_loss(logits, tgt_y, pad_mask);
        
        return std::make_tuple(logits, loss.item<float>());
    }

    // 生成序列（贪婪解码）
    torch::Tensor generate(torch::Tensor src, int64_t max_length = 100) {
        torch::NoGradGuard no_grad;
        
        // 编码源序列
        auto src_mask = mask::create_padding_mask(src);
        torch::Tensor enc_output = encoder_->forward(src, src_mask);
        
        // 初始化目标序列
        auto batch_size = src.size(0);
        auto device = src.device();
        torch::Tensor tgt = torch::zeros({batch_size, 1}, torch::kLong).to(device);
        
        // 逐个生成目标token
        for (int64_t i = 0; i < max_length - 1; ++i) {
            auto tgt_mask = mask::create_combined_mask(tgt);
            
            // 解码当前序列
            auto dec_output = decoder_->forward(tgt, enc_output, tgt_mask, src_mask);
            auto output = projection_->forward(dec_output);
            
            // 选择最可能的下一个token
            auto next_token = output.select(1, -1).argmax(-1, true);
            tgt = torch::cat({tgt, next_token}, 1);
            
            // 检查是否生成了结束标记
            if ((next_token == 2).all().item<bool>()) {  // 假设2是结束标记
                break;
            }
        }
        
        return tgt;
    }

    // Beam Search生成
    torch::Tensor generate_beam(torch::Tensor src, int64_t beam_size = 5, int64_t max_length = 100) {
        torch::NoGradGuard no_grad;
        
        auto batch_size = src.size(0);
        auto device = src.device();
        
        // 编码源序列
        auto src_mask = mask::create_padding_mask(src);
        torch::Tensor enc_output = encoder_->forward(src, src_mask);
        
        // 为beam search准备源编码
        enc_output = enc_output.repeat({beam_size, 1, 1});
        src_mask = src_mask.repeat({beam_size, 1, 1, 1});
        
        // 初始化序列
        std::vector<torch::Tensor> seqs = {torch::zeros({batch_size * beam_size, 1}, torch::kLong).to(device)};
        std::vector<float> scores = {0.0f};
        
        for (int64_t i = 0; i < max_length - 1; ++i) {
            std::vector<torch::Tensor> candidates;
            std::vector<float> candidate_scores;
            
            for (size_t j = 0; j < seqs.size(); ++j) {
                auto seq = seqs[j];
                auto score = scores[j];
                
                auto tgt_mask = mask::create_combined_mask(seq);
                auto dec_output = decoder_->forward(seq, enc_output, tgt_mask, src_mask);
                auto output = projection_->forward(dec_output);
                auto probs = torch::log_softmax(output.select(1, -1), -1);
                
                // 修复topk操作
                auto [values, indices] = probs.topk(beam_size);
                
                for (int64_t k = 0; k < beam_size; ++k) {
                    auto new_seq = torch::cat({seq, indices.select(1, k).unsqueeze(1)}, 1);
                    candidates.push_back(new_seq);
                    candidate_scores.push_back(score + values.select(1, k).item().toFloat());
                }
            }
            
            // 选择最佳的beam_size个候选
            std::vector<size_t> indices(candidate_scores.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + beam_size, indices.end(),
                            [&candidate_scores](size_t i1, size_t i2) {
                                return candidate_scores[i1] > candidate_scores[i2];
                            });
            
            seqs.clear();
            scores.clear();
            for (int64_t k = 0; k < beam_size; ++k) {
                seqs.push_back(candidates[indices[k]]);
                scores.push_back(candidate_scores[indices[k]]);
            }
            
            // 检查是否所有序列都生成了结束标记
            bool all_ended = true;
            for (const auto& seq : seqs) {
                if (seq.select(1, -1).item<int64_t>() != 2) {  // 假设2是结束标记
                    all_ended = false;
                    break;
                }
            }
            if (all_ended) break;
        }
        
        // 返回得分最高的序列
        return seqs[0];
    }

private:
    Encoder encoder_;
    Decoder decoder_;
    torch::nn::Linear projection_;
};

TORCH_MODULE(Transformer); 