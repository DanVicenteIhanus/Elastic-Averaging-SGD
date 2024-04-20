/*
CNN with 3 conv-layers and 1 pooling layer.
Forward pass is implemented here and the backprop is implemented in "main.cpp"
*/
#include "../include/convnet.h"
#include <torch/torch.h>

ConvNetImpl::ConvNetImpl(int64_t num_classes) : fc(16, num_classes) {
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("conv3", conv3);
	register_module("pool", pool);
	register_module("fc", fc);
	}

torch::Tensor ConvNetImpl::forward(torch::Tensor x) {
	x = conv1->forward(x);
	x = conv2->forward(x);
	x = conv3->forward(x);
	x = pool->forward(x);
	x = x.view({-1, 16});
	return fc->forward(x);
}
