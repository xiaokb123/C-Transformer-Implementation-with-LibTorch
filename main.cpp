#include <torch/torch.h>
#include "model/transformer.h"
#include "model/data_loader.h"
#include "model/scheduler.h"
#include <iostream>
#include <vector>
#include <deque>
#include <numeric>
#include <random>

// 早停类
class EarlyStopping {
public:
    EarlyStopping(int patience = 7, float min_delta = 0.0f)
        : patience_(patience), min_delta_(min_delta), counter_(0), best_loss_(std::numeric_limits<float>::infinity()) {}

    bool operator()(float loss) {
        if (loss < best_loss_ - min_delta_) {
            best_loss_ = loss;
            counter_ = 0;
            return false;
        } else {
            counter_++;
            return counter_ >= patience_;
        }
    }

    float get_best_loss() const { return best_loss_; }

private:
    int patience_;
    float min_delta_;
    int counter_;
    float best_loss_;
};

int main() {
    try {
        std::cerr << "Debug: Starting program..." << std::flush << std::endl;
        
        // 检查 LibTorch DLL 路径
        const char* torch_lib_path = std::getenv("PATH");
        if (torch_lib_path) {
            std::cerr << "Debug: PATH environment variable: " << torch_lib_path << std::flush << std::endl;
        } else {
            std::cerr << "Warning: PATH environment variable not found" << std::flush << std::endl;
        }

        // 检查 PyTorch 版本
        std::cerr << "Debug: PyTorch version: " << TORCH_VERSION_MAJOR << "." 
                  << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH 
                  << std::flush << std::endl;

        // 检查 CUDA 可用性
        if (torch::cuda::is_available()) {
            std::cerr << "Debug: CUDA is available" << std::flush << std::endl;
            try {
                torch::Device device(torch::kCUDA);
                std::cerr << "Debug: Created CUDA device" << std::flush << std::endl;
            } catch (const c10::Error& e) {
                std::cerr << "Warning: Failed to create CUDA device: " << e.what() << std::flush << std::endl;
            }
        }

        std::cerr << "Debug: Initializing..." << std::flush << std::endl;
        
        std::cout << "Program starting..." << std::flush << std::endl;
        std::cout << "Torch CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::flush << std::endl;
        std::cout << "Current device: CPU" << std::flush << std::endl;
        
        // 设置模型参数
        const int64_t vocab_size = 10000;  // 词汇表大小
        const int64_t d_model = 512;      // 模型维度
        const int64_t n_heads = 8;        // 注意力头数
        const int64_t n_layers = 6;       // 编码器/解码器层数
        const int64_t d_ff = 2048;        // 前馈网络维度
        const double dropout = 0.1;        // Dropout率

        std::cout << "\nModel parameters:" << std::flush << std::endl
                  << "vocab_size: " << vocab_size << std::flush << std::endl
                  << "d_model: " << d_model << std::flush << std::endl
                  << "n_heads: " << n_heads << std::flush << std::endl
                  << "n_layers: " << n_layers << std::flush << std::endl
                  << "d_ff: " << d_ff << std::flush << std::endl
                  << "dropout: " << dropout << std::flush << std::endl;

        std::cout << "\nTesting individual components..." << std::flush << std::endl;

        // 测试 TokenEmbedding
        std::cout << "\nTesting TokenEmbedding..." << std::flush << std::endl;
        try {
            std::cout << "Creating TokenEmbedding with vocab_size=" << vocab_size << ", d_model=" << d_model << std::flush << std::endl;
            
            // 直接使用 torch::nn::Embedding
            std::cout << "Creating torch::nn::Embedding..." << std::flush << std::endl;
            auto embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, d_model));
            std::cout << "torch::nn::Embedding created successfully" << std::flush << std::endl;

            std::cout << "Creating input tensor..." << std::flush << std::endl;
            torch::Tensor input = torch::randint(0, vocab_size, {2, 10});
            std::cout << "Input tensor created successfully. Shape: " << input.sizes() << std::flush << std::endl;

            std::cout << "Running forward pass..." << std::flush << std::endl;
            torch::Tensor emb_output = embedding->forward(input);
            std::cout << "Forward pass completed. Output shape: " << emb_output.sizes() << std::flush << std::endl;

            // 如果上面的代码成功，再尝试 TokenEmbedding
            std::cout << "\nNow testing TokenEmbedding..." << std::flush << std::endl;
            TokenEmbedding token_embedding(vocab_size, d_model);
            std::cout << "TokenEmbedding created successfully" << std::flush << std::endl;

            torch::Tensor token_output = token_embedding->forward(input);
            std::cout << "TokenEmbedding forward pass completed. Output shape: " << token_output.sizes() << std::flush << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error in TokenEmbedding: " << e.what() << std::endl;
            std::cerr << "Error type: " << typeid(e).name() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "Standard error in TokenEmbedding: " << e.what() << std::endl;
            std::cerr << "Error type: " << typeid(e).name() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "Unknown error in TokenEmbedding" << std::endl;
            throw;
        }

        // 测试 PositionalEncoding
        std::cout << "\nTesting PositionalEncoding..." << std::flush << std::endl;
        try {
            PositionalEncoding pos_enc(d_model, dropout);
            torch::Tensor input = torch::randn({2, 10, d_model});
            torch::Tensor pos_output = pos_enc->forward(input);
            std::cout << "PositionalEncoding test successful. Output shape: " << pos_output.sizes() << std::flush << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error in PositionalEncoding: " << e.what() << std::endl;
            throw;
        }

        // 测试 MultiHeadAttention
        std::cout << "\nTesting MultiHeadAttention..." << std::flush << std::endl;
        try {
            MultiHeadAttention attn(d_model, n_heads);
            torch::Tensor input = torch::randn({2, 10, d_model});
            // 创建布尔类型的掩码
            torch::Tensor mask = torch::zeros({2, 1, 10, 10}, torch::kBool);
            torch::Tensor attn_output = attn->forward(input, input, input, mask);
            std::cout << "MultiHeadAttention test successful. Output shape: " << attn_output.sizes() << std::flush << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error in MultiHeadAttention: " << e.what() << std::endl;
            throw;
        }

        // 测试 EncoderLayer
        std::cout << "\nTesting EncoderLayer..." << std::flush << std::endl;
        try {
            EncoderLayer enc_layer(d_model, n_heads, d_ff, dropout);
            torch::Tensor input = torch::randn({2, 10, d_model});
            torch::Tensor mask = torch::ones({2, 1, 10, 10});
            torch::Tensor enc_output = enc_layer->forward(input, mask);
            std::cout << "EncoderLayer test successful. Output shape: " << enc_output.sizes() << std::flush << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error in EncoderLayer: " << e.what() << std::endl;
            throw;
        }

        // 测试 DecoderLayer
        std::cout << "\nTesting DecoderLayer..." << std::flush << std::endl;
        try {
            DecoderLayer dec_layer(d_model, n_heads, d_ff, dropout);
            torch::Tensor input = torch::randn({2, 8, d_model});
            torch::Tensor enc_output = torch::randn({2, 10, d_model});
            torch::Tensor self_mask = torch::ones({2, 1, 8, 8});
            torch::Tensor enc_mask = torch::ones({2, 1, 8, 10});
            torch::Tensor dec_output = dec_layer->forward(input, enc_output, self_mask, enc_mask);
            std::cout << "DecoderLayer test successful. Output shape: " << dec_output.sizes() << std::flush << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error in DecoderLayer: " << e.what() << std::endl;
            throw;
        }

        // 测试完整的 Encoder
        std::cout << "\nTesting Encoder..." << std::flush << std::endl;
        try {
            Encoder encoder(vocab_size, d_model, n_heads, n_layers, d_ff, dropout);
            torch::Tensor input = torch::randint(0, vocab_size, {2, 10});
            torch::Tensor mask = torch::ones({2, 1, 10, 10});
            torch::Tensor enc_output = encoder->forward(input, mask);
            std::cout << "Encoder test successful. Output shape: " << enc_output.sizes() << std::flush << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error in Encoder: " << e.what() << std::endl;
            throw;
        }

        // 测试完整的 Decoder
        std::cout << "\nTesting Decoder..." << std::flush << std::endl;
        try {
            Decoder decoder(vocab_size, d_model, n_heads, n_layers, d_ff, dropout);
            torch::Tensor input = torch::randint(0, vocab_size, {2, 8});
            torch::Tensor enc_output = torch::randn({2, 10, d_model});
            torch::Tensor self_mask = torch::ones({2, 1, 8, 8});
            torch::Tensor enc_mask = torch::ones({2, 1, 8, 10});
            torch::Tensor dec_output = decoder->forward(input, enc_output, self_mask, enc_mask);
            std::cout << "Decoder test successful. Output shape: " << dec_output.sizes() << std::flush << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error in Decoder: " << e.what() << std::endl;
            throw;
        }

        // 创建完整的 Transformer 模型
        std::cout << "\nCreating Transformer model..." << std::flush << std::endl;
        try {
            Transformer model(vocab_size, d_model, n_heads, n_layers, d_ff, dropout);
            std::cout << "Transformer created successfully" << std::flush << std::endl;
            
            // 确保模型在CPU上
            std::cout << "Moving model to CPU..." << std::flush << std::endl;
            model->to(torch::kCPU);
            std::cout << "Model moved to CPU successfully" << std::flush << std::endl;

            // 创建示例输入数据
            std::cout << "Creating sample input data..." << std::flush << std::endl;
            torch::Tensor src = torch::randint(0, vocab_size, {2, 10});
            torch::Tensor tgt = torch::randint(0, vocab_size, {2, 8});

            std::cout << "Input shapes:" << std::flush << std::endl
                      << "src: " << src.sizes() << std::flush << std::endl
                      << "tgt: " << tgt.sizes() << std::flush << std::endl;

            // 前向传播
            std::cout << "Running forward pass..." << std::flush << std::endl;
            torch::Tensor output = model->forward(src, tgt);
            std::cout << "Forward pass completed successfully" << std::flush << std::endl;
            std::cout << "Output shape: " << output.sizes() << std::flush << std::endl;

            // 创建示例数据
            std::cout << "Creating training data..." << std::flush << std::endl;
            
            // 创建训练数据
            std::vector<std::vector<int64_t>> src_data;
            std::vector<std::vector<int64_t>> tgt_data;
            
            // 生成一些简单的测试数据
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int64_t> dis(1, vocab_size - 1);  // 0 保留给padding
            
            const size_t num_samples = 100;
            const size_t seq_length = 10;
            
            for (size_t i = 0; i < num_samples; ++i) {
                std::vector<int64_t> src(seq_length);
                std::vector<int64_t> tgt(seq_length);
                for (size_t j = 0; j < seq_length; ++j) {
                    src[j] = dis(gen);
                    tgt[j] = dis(gen);
                }
                src_data.push_back(src);
                tgt_data.push_back(tgt);
            }
            
            // 分割数据集
            size_t train_size = 80;  // 80% 用于训练
            std::vector<std::vector<int64_t>> val_src_data(src_data.begin() + train_size, src_data.end());
            std::vector<std::vector<int64_t>> val_tgt_data(tgt_data.begin() + train_size, tgt_data.end());
            src_data.resize(train_size);
            tgt_data.resize(train_size);

            // 创建数据集和数据加载器
            std::cout << "Creating datasets and dataloaders..." << std::flush << std::endl;
            
            auto train_dataset = TransformerDataset(src_data, tgt_data, seq_length)
                .map(torch::data::transforms::Stack<>());
            auto val_dataset = TransformerDataset(val_src_data, val_tgt_data, seq_length)
                .map(torch::data::transforms::Stack<>());

            std::cout << "Train dataset size: " << *train_dataset.size() << std::flush << std::endl;
            std::cout << "Val dataset size: " << *val_dataset.size() << std::flush << std::endl;

            auto train_loader = torch::data::make_data_loader(
                std::move(train_dataset),
                torch::data::DataLoaderOptions().batch_size(8).workers(0)
            );
            auto val_loader = torch::data::make_data_loader(
                std::move(val_dataset),
                torch::data::DataLoaderOptions().batch_size(8).workers(0)
            );
            
            std::cout << "Dataloaders created successfully" << std::flush << std::endl;
            
            // 创建优化器和调度器
            torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));
            TransformerLRScheduler scheduler(optimizer, d_model);

            // 添加初始化输出
            std::cout << "Starting training with parameters:" << std::flush << std::endl
                      << "Vocab size: " << vocab_size << std::flush << std::endl
                      << "Model dimension: " << d_model << std::flush << std::endl
                      << "Number of heads: " << n_heads << std::flush << std::endl
                      << "Number of layers: " << n_layers << std::flush << std::endl
                      << "Feed-forward dimension: " << d_ff << std::flush << std::endl
                      << "Dropout rate: " << dropout << std::flush << std::endl;

            // 创建早停检查器
            EarlyStopping early_stopping(3, 0.001);  // 减少patience
            
            // 训练循环
            int num_epochs = 10;  // 减少epoch数
            std::string checkpoint_path = "transformer_checkpoint.pt";
            float best_val_loss = std::numeric_limits<float>::infinity();

            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                // 训练阶段
                float train_loss = 0.0;
                int train_batches = 0;
                model->train();
                
                for (auto& batch : *train_loader) {
                    auto src = batch.data;
                    auto tgt = batch.target;
                    auto tgt_y = tgt.narrow(1, 1, tgt.size(1) - 1);
                    auto tgt_input = tgt.narrow(1, 0, tgt.size(1) - 1);

                    auto [logits, loss] = model->train_step(src, tgt_input, tgt_y, optimizer);
                    scheduler.step();  // 更新学习率
                    
                    train_loss += loss;
                    train_batches++;

                    if (train_batches % 10 == 0) {
                        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs 
                                  << "], Batch [" << train_batches << "], Loss: " 
                                  << (train_loss / train_batches)
                                  << ", LR: " << scheduler.get_lr() << std::flush << std::endl;
                    }
                }

                // 验证阶段
                float val_loss = 0.0;
                int val_batches = 0;
                model->eval();
                
                for (auto& batch : *val_loader) {
                    torch::NoGradGuard no_grad;
                    
                    auto src = batch.data;
                    auto tgt = batch.target;
                    auto tgt_y = tgt.narrow(1, 1, tgt.size(1) - 1);
                    auto tgt_input = tgt.narrow(1, 0, tgt.size(1) - 1);

                    auto [logits, loss] = model->validate_step(src, tgt_input, tgt_y);
                    val_loss += loss;
                    val_batches++;
                }

                float avg_train_loss = train_loss / train_batches;
                float avg_val_loss = val_loss / val_batches;

                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs 
                          << "], Train Loss: " << avg_train_loss
                          << ", Val Loss: " << avg_val_loss << std::flush << std::endl;

                // 保存最佳模型
                if (avg_val_loss < best_val_loss) {
                    best_val_loss = avg_val_loss;
                    model->save(checkpoint_path);
                    std::cout << "Saved best model with validation loss: " << best_val_loss << std::flush << std::endl;
                }

                // 早停检查
                if (early_stopping(avg_val_loss)) {
                    std::cout << "Early stopping triggered!" << std::flush << std::endl;
                    break;
                }
            }

            // 加载最佳模型进行测试
            model->load(checkpoint_path);
            model->eval();
            
            // 测试推理
            torch::NoGradGuard no_grad;
            
            // 1. 贪婪解码生成
            std::cout << "\n=== Greedy Decoding Generation ===" << std::flush << std::endl;
            torch::Tensor test_src = torch::randint(0, vocab_size, {1, 10});
            torch::Tensor generated = model->generate(test_src);
            std::cout << "Input shape: " << test_src.sizes() << std::flush << std::endl;
            std::cout << "Generated sequence shape: " << generated.sizes() << std::flush << std::endl;
            
            // 2. Beam Search生成
            std::cout << "\n=== Beam Search Generation ===" << std::flush << std::endl;
            torch::Tensor beam_generated = model->generate_beam(test_src, 5);
            std::cout << "Beam search generated shape: " << beam_generated.sizes() << std::flush << std::endl;
            
            // 比较两种生成方法的结果
            std::cout << "\n=== Generation Results ===" << std::flush << std::endl;
            std::cout << "Input sequence: " << test_src << std::flush << std::endl;
            std::cout << "Greedy generated: " << generated << std::flush << std::endl;
            std::cout << "Beam search generated: " << beam_generated << std::flush << std::endl;

            return 0;
        } catch (const c10::Error& e) {
            std::cerr << "Error in Transformer: " << e.what() << std::flush << std::endl;
            throw;
        }
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error occurred: " << e.what() << std::flush << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error occurred: " << e.what() << std::flush << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::flush << std::endl;
        return 1;
    }
}