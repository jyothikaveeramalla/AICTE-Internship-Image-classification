# AICTE-Internship-Image-classification
This project implements an image classification system utilizing the MobileNetV2 architecture and the CIFAR-10 dataset. The goal is to train a lightweight and efficient deep learning model capable of accurately classifying images into ten categories.

Table of Contents

Introduction

Features

Dataset

Model Architecture

Prerequisites

Installation

Usage

Results

Future Work

Contributing

License

Introduction

Image classification is a fundamental task in computer vision. This project uses the CIFAR-10 dataset, consisting of 60,000 32x32 color images divided into 10 classes. The MobileNetV2 model, a lightweight and efficient convolutional neural network, is employed to balance computational efficiency with classification accuracy.

Features

Preprocessing pipeline for the CIFAR-10 dataset.

Transfer learning using MobileNetV2.

Custom training loop with metrics for evaluation.

Easy-to-use Streamlit app for real-time predictions.

Dataset

The CIFAR-10 dataset contains 10 classes:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

The dataset is split into 50,000 training images and 10,000 test images.

Model Architecture

MobileNetV2 is a lightweight deep neural network designed for mobile and embedded vision applications. Key features include:

Depthwise separable convolutions.

Inverted residuals with linear bottlenecks.

Tunable width multiplier for balancing efficiency and accuracy.

For this project:

The base MobileNetV2 is pretrained on ImageNet.

A custom dense layer is added to adapt the model to CIFAR-10.

Fine-tuning is performed on the CIFAR-10 dataset.

Prerequisites

Python 3.8 or higher

TensorFlow 2.10 or higher

Streamlit 1.18 or higher

Installation

Clone the repository:

git clone https://github.com/jyothikaveeramalla/AICTE-Internship-Image-classification

Create a virtual environment:

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Usage

Training the Model

Prepare the CIFAR-10 dataset (automatically handled by the script).

Run the training script:

python train.py

The trained model will be saved in the models directory.

Running the Streamlit App

Launch the app:

streamlit run app.py

Upload an image and see the classification results in real time.

Results

After training for [N] epochs, the model achieved:

Training accuracy: [X]%

Validation accuracy: [Y]%

Example predictions and evaluation metrics are provided in the results directory.

Future Work

Experiment with other architectures like ResNet or EfficientNet.

Improve preprocessing techniques for better generalization.

Extend the app to support batch predictions.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
