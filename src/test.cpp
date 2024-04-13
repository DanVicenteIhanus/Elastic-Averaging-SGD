#include <torch/torch.h>

class ConvNetImpl : public torch::nn::Module {
public:
    explicit ConvNetImpl(int64_t num_classes = 10) {
        // Initialize modules here and register them
        layer1 = register_module("layer1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)),
            torch::nn::BatchNorm2d(16),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        ));
        layer2 = register_module("layer2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        ));
        fc = register_module("fc", torch::nn::Linear(32 * whatever_size_here, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = x.view({-1, 32 * whatever_size_here}); // Flatten
        x = fc->forward(x);
        return x;
    }

private:
    torch::nn::Sequential layer1, layer2;
    torch::nn::Linear fc;
};

TORCH_MODULE(ConvNet);  // Wraps ConvNetImpl into ConvNet that handles shared pointers