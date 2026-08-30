import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class CreditRiskNN:
    def __init__(self, input_size=3, hidden_size=5, output_size=1):
        # W1 connects 3 input features to 5 hidden neurons (3x5 matrix)
        self.W1 = np.random.normal(size=(input_size, hidden_size))
        self.b1 = np.random.normal(size=(1, hidden_size))

        # W2 connects 5 hidden neurons to 1 output (5x1 matrix)
        self.W2 = np.random.normal(size=(hidden_size, output_size))
        self.b2 = np.random.normal(size=(1, output_size))

    def feedforward(self, X):
        # Matrix dot products calculate all neurons and samples simultaneously
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1500

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Reshape arrays into matrices (1xN) for dot product math
                x = x.reshape(1, -1)
                y = np.array([[y_true]])

                # --- Feedforward ---
                Z1 = np.dot(x, self.W1) + self.b1
                A1 = sigmoid(Z1)
                Z2 = np.dot(A1, self.W2) + self.b2
                y_pred = sigmoid(Z2)

                # --- Backpropagation ---
                # 1. Output Layer Gradients
                dL_dypred = -2 * (y - y_pred)
                dypred_dZ2 = deriv_sigmoid(Z2)
                dL_dZ2 = dL_dypred * dypred_dZ2

                dL_dW2 = np.dot(A1.T, dL_dZ2)
                dL_db2 = dL_dZ2

                # 2. Hidden Layer Gradients (Propagating the error backwards)
                dL_dA1 = np.dot(dL_dZ2, self.W2.T)
                dA1_dZ1 = deriv_sigmoid(Z1)
                dL_dZ1 = dL_dA1 * dA1_dZ1

                dL_dW1 = np.dot(x.T, dL_dZ1)
                dL_db1 = dL_dZ1

                # --- Update weights and biases ---
                self.W1 -= learn_rate * dL_dW1
                self.b1 -= learn_rate * dL_db1
                self.W2 -= learn_rate * dL_dW2
                self.b2 -= learn_rate * dL_db2

            # Calculate and print total loss every 100 epochs
            if epoch % 100 == 0:
                y_preds = self.feedforward(data)
                loss = mse_loss(all_y_trues.reshape(-1, 1), y_preds)
                print(f"Epoch {epoch} loss: {loss:.4f}")

# --- Dataset: Credit Risk Assessment ---
# Inputs: [Credit Score (normalized), Debt-to-Income Ratio, Loan-to-Value Ratio]
# Output: 1 (High Risk / Default), 0 (Low Risk / Repaid)
data = np.array([
    [ 0.8,  0.2,  0.3],  # Excellent credit, low debt, low loan value -> 0 (Repaid)
    [-0.9,  0.8,  0.9],  # Poor credit, high debt, high loan value -> 1 (Default)
    [ 0.5,  0.4,  0.5],  # Average credit, moderate debt, moderate loan -> 0 (Repaid)
    [-0.6,  0.7,  0.4],  # Below average credit, high debt, moderate loan -> 1 (Default)
    [ 0.9,  0.1,  0.8],  # Excellent credit, very low debt, high loan value -> 0 (Repaid)
])

all_y_trues = np.array([0, 1, 0, 1, 0])

# Initialize and train
network = CreditRiskNN(input_size=3, hidden_size=5, output_size=1)
network.train(data, all_y_trues)
