#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "Program started" << std::endl;
    std::cout.flush();
    
    try {
        std::cout << "Starting LibTorch test program..." << std::endl;
        std::cout.flush();
        
        // 检查 CUDA 可用性
        std::cout << "Checking CUDA availability..." << std::endl;
        std::cout.flush();
        std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
        std::cout.flush();
        
        // 创建一个简单的张量
        std::cout << "Creating test tensor..." << std::endl;
        std::cout.flush();
        torch::Tensor tensor = torch::rand({2, 3});
        std::cout << "Created tensor:\n" << tensor << std::endl;
        std::cout.flush();
        
        // 创建一个简单的神经网络层
        std::cout << "Creating linear layer..." << std::endl;
        std::cout.flush();
        torch::nn::Linear linear(10, 5);
        std::cout << "Linear layer created successfully" << std::endl;
        std::cout.flush();
        
        // 测试前向传播
        std::cout << "Testing forward pass..." << std::endl;
        std::cout.flush();
        torch::Tensor input = torch::randn({3, 10});
        std::cout << "Input tensor size: " << input.sizes() << std::endl;
        std::cout.flush();
        
        torch::Tensor output = linear->forward(input);
        std::cout << "Forward pass successful" << std::endl;
        std::cout.flush();
        std::cout << "Output tensor size: " << output.sizes() << std::endl;
        std::cout.flush();
        
        std::cout << "Test completed successfully!" << std::endl;
        std::cout.flush();
        return 0;
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        std::cerr.flush();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        std::cerr.flush();
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        std::cerr.flush();
        return 1;
    }
} 