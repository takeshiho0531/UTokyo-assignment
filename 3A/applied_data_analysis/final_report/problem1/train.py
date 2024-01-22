import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DNN import RegressionModel


def generate_data(N, epsilon, x_range=(0, 100)):
    x = torch.FloatTensor(N).uniform_(x_range).unsqueeze(1)
    y = 5 * x * torch.sin(2 * np.pi * x) + 4 * torch.exp(1 / (x + 1)) + epsilon
    return x, y


def train(x, y, epochs, hidden_layer1, hidden_layer2, dropout_ratio, lr):
    model = RegressionModel(
        hidden_layer1=hidden_layer1,
        hidden_layer2=hidden_layer2,
        dropout_ratio=dropout_ratio,
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


def main(
    N: int,
    epsilon: float,
    x_range,
    epochs: int,
    hidden_layer1: int,
    hidden_layer2: int,
    dropout_ratio: float,
    lr: float,
):
    x, y = generate_data(N=N, epsilon=epsilon, x_range=x_range)
    train(
        x=x,
        y=y,
        epochs=epochs,
        hidden_layer1=hidden_layer1,
        hidden_layer2=hidden_layer2,
        dropout_ratio=dropout_ratio,
        lr=lr,
    )


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        sys.argv[7],
        sys.argv[8],
    )
