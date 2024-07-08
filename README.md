# CIFAR-10 Image Classification

This repository contains implementations of two different neural network architectures to classify images from the CIFAR-10 dataset:

- **Convolutional Neural Network (CNN)** - (`cifar10_cnn.ipynb`)
- **Multi-Layer Perceptron (MLP)** - (`cifar10_mlp.ipynb`)

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

## Requirements
To run the notebooks, install the required dependencies using:

```bash
pip install keras tensorflow numpy matplotlib
```

## Notebooks
### 1. CNN Model (`cifar10_cnn.ipynb`)
This notebook implements a **Convolutional Neural Network (CNN)** to classify images. The CNN model includes:
- Multiple convolutional layers
- Max pooling layers
- Fully connected layers
- Softmax classifier

### 2. MLP Model (`cifar10_mlp.ipynb`)
This notebook implements a **Multi-Layer Perceptron (MLP)** to classify images. The MLP model includes:
- Fully connected dense layers
- Activation functions (ReLU, Softmax)
- Dropout layers for regularization

## Running the Notebooks
To run the notebooks, execute the following command:

```bash
jupyter notebook
```

Then open `cifar10_cnn.ipynb` or `cifar10_mlp.ipynb` in Jupyter Notebook and run the cells.

## Results
The CNN model is expected to achieve significantly better accuracy compared to the MLP model, as CNNs are more effective for image classification tasks.
