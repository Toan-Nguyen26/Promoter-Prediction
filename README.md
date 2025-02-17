# Promoter Sequence Classifier

A deep learning model for classifying bacterial promoter sequences using Convolutional Neural Networks (CNN). This project implements a binary classifier to distinguish between promoter and non-promoter DNA sequences like in E. coli.

## Overview

The model uses a CNN architecture specifically designed for sequence data, with the following key features:
- One-hot encoding for DNA sequence representation
- Convolutional layers for feature extraction
- Batch normalization and dropout for regularization
- Focal Loss for handling class imbalance

## Dataset

The dataset used in this project is from [CNNPromoterData](https://github.com/solovictor/CNNPromoterData/tree/master), which was originally used in the paper: