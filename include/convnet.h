#include <torch/torch.h>
#include <iostream>

class ConvNetImpl : public torch::nn::Module {
	public:
		explicit ConvNetImpl(int64_t num_classes = 10);
		torch::Tensor forward(torch::Tensor x);

	private:
		torch::nn::Sequential conv1{
			torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(16),
			torch::nn::ReLU(),
			torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
		};

		torch::nn::Sequential conv2{
			torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(32),
			torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
		};

		torch::nn::Sequential conv3{
			torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(64),
			torch::nn::ReLU()
		};

		// adaptive average pooling and fully connected final layer
		torch::nn::AdaptiveAvgPool2d pool{torch::nn::AdaptiveAvgPool2dOptions({1, 1})};
		torch::nn::Linear fc;
	};

	TORCH_MODULE(ConvNet);


