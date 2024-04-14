#include <torch/torch.h>
/*
TODO: 
1. Define the convnet (IMPL is added for pytorch to understand this as a module)
2. Think through forward pass and dimensions
3. 
*/
class ConvNetImpl : public torch::nn::Module {
    public:
        explicit ConvNetImpl(int64_t num_classes = 10) {
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
            fc = register_module("fc", torch::nn::Linear(32 * 1, num_classes));
        }

        torch::Tensor forward(torch::Tensor x) {
            x = layer1->forward(x);
            x = layer2->forward(x);
            x = x.view({-1, 32 * 1}); // Flatten
            x = fc->forward(x);
            return x;
        }

    private:
        torch::nn::Sequential layer1, layer2;
        torch::nn::Linear fc;
};

TORCH_MODULE(ConvNet); 