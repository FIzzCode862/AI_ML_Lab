# CNN-Based Fashion Image Classification

## Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification using the FashionMNIST dataset.

The trained CNN classifies images into 10 FashionMNIST categories and is also tested on 10 real-world smartphone photographs.

The project is implemented using Python and PyTorch in Google Colab.

---

## Dataset

The project uses the **FashionMNIST** dataset.

FashionMNIST contains 10 classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset is automatically downloaded using `torchvision`.

---

## Preprocessing

The images are processed using the following steps:

- Resize images to `28 × 28`
- Convert images to tensors
- Normalize pixel values

The same preprocessing pipeline is used when testing the custom smartphone images.

---

## CNN Architecture

The CNN consists of:

```text
Input Image
    ↓
Conv2D (1 → 32)
    ↓
ReLU
    ↓
MaxPool2D
    ↓
Conv2D (32 → 64)
    ↓
ReLU
    ↓
MaxPool2D
    ↓
Flatten
    ↓
Fully Connected Layer (3136 → 128)
    ↓
ReLU
    ↓
Fully Connected Layer (128 → 10)
    ↓
Output


## Real-World Prediction Analysis

The CNN was tested on 10 smartphone photographs. Some predictions were incorrect because the real-world images differ significantly from the FashionMNIST training images.

FashionMNIST contains simple 28×28 grayscale images, while smartphone photographs contain backgrounds, lighting variations, colors, shadows, and additional visual noise.

Therefore, the real-world prediction results demonstrate the limitation of applying a model trained on FashionMNIST directly to real-world photographs.

