#include <torch/torch.h>
#include <iostream>

class ConvNetImpl : public torch::nn::Module {
	public:
		explicit ConvNetImpl(int64_t num_classes = 10);
		torch::Tensor forward(torch::Tensor x);

	private:
		torch::nn::Sequential conv1 {
			torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 30, 5).stride(1)),
			// add padding by .padding(1) (or whatever number < 28, instead of 1)
			torch::nn::ReLU(),
			torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
		};

		torch::nn::Sequential conv2 {
			torch::nn::Conv2d(torch::nn::Conv2dOptions(30, 40, 5).stride(1)),
			torch::nn::ReLU(),
			torch::nn::BatchNorm2d(40),
			torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
		};

		torch::nn::Sequential fc1 {
			torch::nn::Linear(torch::nn::LinearOptions(4*4*40, 700)),
			torch::nn::ReLU(),
			//torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.3)) // too much dropout?
		};

		torch::nn::Sequential fc2 {
			torch::nn::Linear(torch::nn::LinearOptions(700, 10))
		};

		// adaptive average pooling and fully connected final layers
		//torch::nn::AdaptiveAvgPool2d pool{torch::nn::AdaptiveAvgPool2dOptions({1, 1})};	
	};

	TORCH_MODULE(ConvNet);


