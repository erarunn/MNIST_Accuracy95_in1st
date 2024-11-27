# MNIST Digit Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Passing-success)
![Build Status](https://github.com/erarunn/MNIST_Accuracy95_in1st/actions/workflows/model_tests.yml/badge.svg)

A lightweight CNN model for MNIST digit classification that achieves >95% accuracy in just one epoch, with less than 25k parameters.

## 🎯 Model Performance Requirements

- [x] Parameters < 25,000
- [x] Accuracy > 95% in 1 epoch
- [x] Automated testing via GitHub Actions

## 🧪 Tests

The model undergoes several automated tests:
1. Parameter count verification (< 25k)
2. Accuracy validation (> 95%)
3. Model prediction shape validation
4. Gradient flow verification
5. Batch processing capability
6. Image augmentation verification

## 🖼️ Data Augmentation

The model uses various augmentation techniques:
- Random rotation (±15 degrees)
- Random translation (±10%)
- Normalization

Sample augmented images can be found in the `augmentation_samples` directory.

## 🏗️ Architecture

The model uses a carefully designed CNN architecture with:
- Multiple convolutional layers with batch normalization
- Strategic dropout for regularization
- MaxPooling for spatial dimension reduction
- Efficient parameter usage through proper channel management

## 📊 Model Summary 