import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DNN import RegressionModel


def generate_data(N, epsilon, x_range=(0, 100)):
    x = torch.FloatTensor(N).uniform_(x_range).unsqueeze(1)
    y = 5 * x * torch.sin(2 * np.pi * x) + 4 * torch.exp(1 / (x + 1)) + epsilon
    return x, y


N = 10

x, y = generate_data(N)


epochs = 10


def train(epoch, x, y):
    model = RegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
