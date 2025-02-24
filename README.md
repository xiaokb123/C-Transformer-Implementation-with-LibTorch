# C++ Transformer Implementation with LibTorch

This project implements a Transformer model using C++ and LibTorch (PyTorch's C++ API). The implementation includes both encoder and decoder architectures, attention mechanisms, and all the essential components of the Transformer architecture as described in the paper "Attention Is All You Need".

## Features

- Complete Transformer architecture implementation in C++
- Modular design with separate components (encoder, decoder, attention, etc.)
- Support for both training and inference
- Beam search and greedy decoding for sequence generation
- Custom learning rate scheduler
- Early stopping mechanism
- Configurable model parameters
- Data loading and batching utilities

## Prerequisites

- C++17 compatible compiler
- CMake (version 3.0 or higher)
- LibTorch (PyTorch C++ library) version 2.6.0
- Visual Studio 2022 (for Windows) or equivalent C++ development environment
- CUDA toolkit (optional, for GPU support)

## Installation

1. Download LibTorch from the official PyTorch website:
   ```bash
   # CPU version (Debug)
   https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.6.0%2Bcpu.zip
   ```

2. Extract LibTorch to a known location (e.g., `D:/libtorch`)

3. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cpp-transformer.git
   cd cpp-transformer
   ```

4. Configure the project:
   ```bash
   mkdir build
   cd build
   cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=path/to/libtorch ..
   ```

5. Build the project:
   ```bash
   cmake --build . --config Debug
   ```

## Project Structure

### Core Components

- `main.cpp`: Main application entry point and training loop
- `test.cpp`: Simple test program to verify LibTorch functionality

### Model Components (`model/` directory)

- `transformer.h`: Complete Transformer model implementation
- `encoder.h/cpp`: Transformer encoder implementation
- `decoder.h/cpp`: Transformer decoder implementation
- `attention.h/cpp`: Multi-head attention mechanism
- `embedding.h/cpp`: Token and positional embeddings
- `mask.h`: Masking utilities for attention
- `scheduler.h`: Learning rate scheduler implementation
- `data_loader.h`: Data loading and processing utilities

### Build Configuration

- `CMakeLists.txt`: CMake build configuration
- `.vscode/`: VSCode configuration files
  - `launch.json`: Debugging configuration
  - `tasks.json`: Build tasks
  - `c_cpp_properties.json`: C++ configuration

## Usage

1. Configure model parameters in `main.cpp`:
   ```cpp
   const int64_t vocab_size = 10000;
   const int64_t d_model = 512;
   const int64_t n_heads = 8;
   const int64_t n_layers = 6;
   const int64_t d_ff = 2048;
   const double dropout = 0.1;
   ```

2. Run the test program to verify setup:
   ```bash
   ./build/Debug/test_torch.exe
   ```

3. Run the main program:
   ```bash
   ./build/Debug/transformer.exe
   ```

## Model Architecture

The implementation follows the original Transformer architecture with:

- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Positional encoding
- Token embeddings

## Training

The model includes:

- Custom learning rate scheduler with warmup
- Early stopping mechanism
- Beam search for inference
- Cross-entropy loss function
- Gradient clipping
- Support for both training and validation phases

## Debugging

The project includes comprehensive debugging support:

- Detailed logging
- Visual Studio debugger integration
- Memory leak detection
- Exception handling
- Performance profiling support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- "Attention Is All You Need" paper authors
- PyTorch team for LibTorch
- C++ community

## Contact

For questions and feedback, please open an issue in the GitHub repository. 