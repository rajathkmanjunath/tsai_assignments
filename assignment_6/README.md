# MNIST Classification Model
## Overview
This project implements a CNN model for MNIST digit classification with the following constraints:
- Validation accuracy ≥ 99.4%
- Number of parameters < 20K

## Model Architecture
The model uses:
- 3 convolutional layers with batch normalization
- MaxPooling layers
- Single fully connected layer
- Total parameters: ~18K

## Usage
1. Install dependencies: 
```bash
pip install torch torchvision tqdm
```
2. Train and evaluate the model:
```bash
python train.py
```
## Features
- Progress bars showing training and evaluation progress
- Detailed logging of training metrics
- Automatic saving of best model
- Training log file creation
- Real-time accuracy and loss monitoring

## Model Performance
- Validation Accuracy: ~99.4%
- Total Parameters: <20K

## Logs and Artifacts
The training process creates:
- `training.log`: Detailed training logs
- `best_model.pth`: Best model weights
- Real-time progress bars during training

## GitHub Actions
The workflow automatically:
1. Validates that model achieves ≥99.4% accuracy on the test set
2. Confirms model has <20K parameters
3. Uploads training logs and best model as artifacts

[![Assignment 6](https://github.com/rajathkmanjunath/tsai_assignments/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/rajathkmanjunath/tsai_assignments/actions/workflows/assignment_6.yml)