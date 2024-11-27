# Machine Learning Pipeline

This project implements an automated machine learning pipeline for training and testing a model on the MNIST dataset.

## Project Structure


├── src/    
│ └── train.py  
├── tests/  
├── models/     
├── requirements.txt    
└── README.md   


## Overview

This project uses GitHub Actions to automatically:
1. Set up a Python 3.8 environment
2. Install dependencies
3. Train the model
4. Run unit tests
5. Save the trained model as an artifact

## Requirements

- Python 3.8
- Dependencies listed in `requirements.txt`

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed for MNIST digit classification:

- Input Layer: 28x28 grayscale images
- Convolutional Layer 1: 4 filters (3x3), ReLU activation
- MaxPooling Layer 1: 2x2 pool size
- Convolutional Layer 2: 8 filters (3x3), ReLU activation
- MaxPooling Layer 2: 2x2 pool size
- Convolutional Layer 3: 4 filters (1x1)
- Convolutional Layer 4: 16 filters (3x3), ReLU activation
- MaxPooling Layer 3: 2x2 pool size
- Flatten Layer
- Dense Layer 1: 144 units
- Output Layer: 10 units (one per digit), Softmax activation


## Usage

### Local Development

1. Clone the repository
2. Navigate to the assignment_5 directory
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run training:
```bash
python src/train.py
```
5. Run tests:
```bash
python -m unittest discover tests
```

### CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically runs the entire pipeline on every push to the repository. The trained model is saved as an artifact and can be downloaded from the GitHub Actions page.

## Model Artifacts

The trained model is saved as `model_mnist_latest.pth` in the `models/` directory.

## Testing

Tests are located in the `tests/` directory and are automatically run as part of the CI/CD pipeline.

# Your Project Name

[![Assignment 5](https://github.com/rajathkmanjunath/tsai_assignments/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/rajathkmanjunath/tsai_assignments/actions/workflows/ml-pipeline.yml)