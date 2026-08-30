# Credit Risk Assessment Neural Network

## Overview
This repository contains a lightweight, educational Feedforward Neural Network built entirely from scratch using standard Python and NumPy. It is designed to predict credit default risk based on standardized financial metrics. 

By avoiding high-level libraries like TensorFlow or PyTorch, this project demonstrates the fundamental mathematics powering modern deep learning, specifically **vectorization** and **backpropagation**.

## Architecture
The network follows a classic Multilayer Perceptron (MLP) structure:
* **Input Layer:** 3 neurons representing normalized financial features:
  1. Credit Score
  2. Debt-to-Income (DTI) Ratio
  3. Loan-to-Value (LTV) Ratio
* **Hidden Layer:** 5 neurons utilizing the Sigmoid activation function to capture non-linear relationships.
* **Output Layer:** 1 neuron (Sigmoid activation) outputting a continuous probability between 0 and 1, representing the risk of loan default.

## Key Features
* **Vectorized Operations:** Replaces inefficient scalar variables with NumPy matrix dot products, significantly speeding up feedforward and backpropagation phases.
* **Gradient Descent:** Implements the chain rule of calculus to calculate partial derivatives and update network weights/biases iteratively.
* **Mean Squared Error (MSE):** Utilizes standard MSE for the loss function to quantify the network's predictive accuracy over time.

## Installation & Usage
### Prerequisites
* Python 3.6+
* NumPy

### Installation
Clone this repository and ensure NumPy is installed:
```bash
git clone https://github.com/yourusername/credit-risk-nn.git
cd credit-risk-nn
pip install numpy
```

### Running the Model
To train the model and view the loss reduction over 1500 epochs, run:
```bash
python credit_risk_nn.py
```
*Expected Output:* You should see the Mean Squared Error (MSE) loss steadily decreasing every 100 epochs, demonstrating that the network is successfully learning the underlying patterns in the dataset.

## The Dataset
The model is trained on a synthetic dataset reflecting standard risk assessment paradigms:
* `1` indicates High Risk (Default).
* `0` indicates Low Risk (Repaid).

*Example input:* `[0.8, 0.2, 0.3]` (High credit, low debt, low loan value) correlates to a `0` (Repaid).

## Technical Deep Dive: Why Vectorization?
In standard procedural programming, calculating weights for a multi-neuron network requires nested `for` loops. By structuring weights as matrices ($W_1$ as a $3 	imes 5$ matrix and $W_2$ as a $5 	imes 1$ matrix), we compute all neuron activations simultaneously using linear algebra (`np.dot(X, W)`). This is the exact same architectural principle that allows modern neural networks to scale across GPUs.
