#pragma once
#include <torch/torch.h>
#include <cmath>

class TransformerLRScheduler {
public:
    TransformerLRScheduler(torch::optim::Optimizer& optimizer,
                          int64_t d_model,
                          int64_t warmup_steps = 4000)
        : optimizer_(optimizer),
          d_model_(d_model),
          warmup_steps_(warmup_steps),
          step_num_(0) {}

    void step() {
        ++step_num_;
        float lr = compute_lr();
        for (auto& group : optimizer_.param_groups()) {
            group.options().set_lr(lr);
        }
    }

    float get_lr() const {
        return compute_lr();
    }

private:
    float compute_lr() const {
        float step_num = static_cast<float>(step_num_);
        float d_model = static_cast<float>(d_model_);
        float warmup_steps = static_cast<float>(warmup_steps_);
        
        float arg1 = 1.0f / std::sqrt(step_num);
        float arg2 = step_num * (warmup_steps * warmup_steps);
        
        return std::sqrt(d_model) * std::min(arg1, arg2);
    }

    torch::optim::Optimizer& optimizer_;
    int64_t d_model_;
    int64_t warmup_steps_;
    int64_t step_num_;
}; 