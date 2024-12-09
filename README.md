---
title: "Image Classification on CIFAR-10 with ResNet-152 and EfficientNet-B2"
output: 
  github_document: default
---


# Image Classification on CIFAR-10 with ResNet-152 and EfficientNet-B2

This repository contains code for fine-tuning and evaluating two state-of-the-art deep learning models, ResNet-152 and EfficientNet-B2, on the CIFAR-10 dataset. The project compares the performance of these models in terms of classification accuracy, computational efficiency, and generalization ability.

## Dataset

The [CIFAR-10 dataset](https://huggingface.co/datasets/uoft-cs/cifar10) consists of 60,000 32x32 color images divided into 10 classes, with 50,000 images for training and 10,000 for testing.

## Models

- **[ResNet-152](https://huggingface.co/microsoft/resnet-152):** A deep residual network introduced to mitigate the vanishing gradient problem by using skip connections.
- **[EfficientNet-B2](https://huggingface.co/google/efficientnet-b2):** A scalable convolutional network architecture that optimizes accuracy and computational efficiency through compound scaling.

## Files

- **[`fine_tune_resnet_152.ipynb`](./fine_tune_resnet_152.ipynb):** Jupyter Notebook for fine-tuning and evaluating the ResNet-152 model on CIFAR-10.
- **[`fine_tune_efficientnet_b2.ipynb`](./fine_tune_efficientnet_b2.ipynb):** Jupyter Notebook for fine-tuning and evaluating the EfficientNet-B2 model on CIFAR-10.

## Installation

To run the code, ensure you have Python installed along with the following libraries:

```bash
pip install transformers datasets torch torchvision matplotlib scikit-learn
