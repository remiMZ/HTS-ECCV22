# Backbones

Several backbone networks are provided:

* `con4.py`: Conv-4
* `resnet12.py`: ResNet-12

## Introduction

### Conv-4

This architecture is composed of 4 convolutional blocks. Each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity and a 2x2 max-pooling layer.

### ResNet-12

Follwing the paper [1]。

## Reference

[1] [few-shot-meta-baseline](https://github.com/cyvius96/few-shot-meta-baseline/blob/master/models/resnet12.py)