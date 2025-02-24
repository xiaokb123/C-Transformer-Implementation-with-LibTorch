#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>

/**
 * @brief Transformer数据集类
 * 用于加载和处理源序列和目标序列数据
 */
class TransformerDataset : public torch::data::Dataset<TransformerDataset> {
public:
    /**
     * @brief 构造函数
     * @param src_data 源序列数据
     * @param tgt_data 目标序列数据
     * @param max_length 最大序列长度
     * @param pad_idx 填充token的索引值
     */
    TransformerDataset(const std::vector<std::vector<int64_t>>& src_data,
                      const std::vector<std::vector<int64_t>>& tgt_data,
                      int64_t max_length = 100,
                      int64_t pad_idx = 0) 
        : src_data_(src_data), tgt_data_(tgt_data), 
          max_length_(max_length), pad_idx_(pad_idx) {
        if (src_data.size() != tgt_data.size()) {
            throw std::runtime_error("Source and target data size mismatch");
        }
        size_ = src_data.size();
    }

    /**
     * @brief 获取指定索引的数据样本
     * @param index 样本索引
     * @return 包含源序列和目标序列的样本对
     */
    torch::data::Example<> get(size_t index) override {
        std::vector<int64_t> src = src_data_[index];
        std::vector<int64_t> tgt = tgt_data_[index];

        // 截断序列
        if (src.size() > max_length_) src.resize(max_length_);
        if (tgt.size() > max_length_) tgt.resize(max_length_);

        // 填充序列
        src = pad_sequence(src);
        tgt = pad_sequence(tgt);

        // 转换为tensor
        torch::Tensor src_tensor = torch::from_blob(src.data(), {max_length_}, torch::kLong).clone();
        torch::Tensor tgt_tensor = torch::from_blob(tgt.data(), {max_length_}, torch::kLong).clone();
        
        return {src_tensor, tgt_tensor};
    }

    /**
     * @brief 获取数据集大小
     * @return 数据集中的样本数量
     */
    torch::optional<size_t> size() const override {
        return size_;
    }

    /**
     * @brief 创建一个小的测试数据集
     * @param vocab_size 词汇表大小
     * @param num_samples 样本数量
     * @param seq_length 序列长度
     * @return 包含随机生成的源序列和目标序列的数据集
     */
    static TransformerDataset create_test_dataset(
        int64_t vocab_size,
        size_t num_samples = 100,
        int64_t seq_length = 10
    ) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dis(1, vocab_size - 1); // 0通常保留给padding

        std::vector<std::vector<int64_t>> src_data;
        std::vector<std::vector<int64_t>> tgt_data;

        for (size_t i = 0; i < num_samples; ++i) {
            std::vector<int64_t> src(seq_length);
            std::vector<int64_t> tgt(seq_length);
            for (int64_t j = 0; j < seq_length; ++j) {
                src[j] = dis(gen);
                tgt[j] = dis(gen);
            }
            src_data.push_back(src);
            tgt_data.push_back(tgt);
        }

        return TransformerDataset(src_data, tgt_data, seq_length);
    }

private:
    std::vector<std::vector<int64_t>> src_data_;
    std::vector<std::vector<int64_t>> tgt_data_;
    size_t size_;
    int64_t max_length_;
    int64_t pad_idx_;

    /**
     * @brief 对序列进行填充
     * @param seq 输入序列
     * @return 填充后的序列
     */
    std::vector<int64_t> pad_sequence(const std::vector<int64_t>& seq) const {
        std::vector<int64_t> padded = seq;
        while (padded.size() < max_length_) {
            padded.push_back(pad_idx_);
        }
        return padded;
    }
}; 