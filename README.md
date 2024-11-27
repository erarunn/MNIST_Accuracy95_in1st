# MNIST Digit Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Passing-success)

A lightweight CNN model for MNIST digit classification that achieves >95% accuracy in just one epoch, with less than 25k parameters.

## 🎯 Model Performance Requirements

- [x] Parameters < 25,000
- [x] Accuracy > 95% in 1 epoch
- [x] Automated testing via GitHub Actions

## 🏗️ Architecture

The model uses a carefully designed CNN architecture with:
- Multiple convolutional layers with batch normalization
- Strategic dropout for regularization
- MaxPooling for spatial dimension reduction
- Efficient parameter usage through proper channel management

## 📊 Model Summary 