import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DNN import RegressionDeeperModel, RegressionModel


def generate_data(N, epsilon, x_range_from=0, x_range_to=1):
    x = torch.FloatTensor(N).uniform_(x_range_from, x_range_to).unsqueeze(1)
    y = 5 * x * torch.sin(2 * np.pi * x) + 4 * torch.exp(1 / (x + 1)) + epsilon
    return x, y


def train(x, y, save_path, epochs, hidden_layer1, hidden_layer2, dropout_ratio, lr):
    loss_list = []
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

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        loss_list.append(loss.item())

    torch.save(model.state_dict(), save_path)
    return loss_list


def train_wrapper(
    N: int,
    epsilon: float,
    x_range_from: float,
    x_range_to: float,
    save_path: str,
    epochs: int,
    hidden_layer1: int,
    hidden_layer2: int,
    dropout_ratio: float,
    lr: float,
) -> List[float]:
    x, y = generate_data(
        N=N, epsilon=epsilon, x_range_from=x_range_from, x_range_to=x_range_to
    )
    loss_list = train(
        x=x,
        y=y,
        save_path=save_path,
        epochs=epochs,
        hidden_layer1=hidden_layer1,
        hidden_layer2=hidden_layer2,
        dropout_ratio=dropout_ratio,
        lr=lr,
    )
    return loss_list


def train_deeper_model_wrapper(
    N: int,
    epsilon: float,
    x_range_from: float,
    x_range_to: float,
    save_path: str,
    epochs: int,
    hidden_layer1: int,
    hidden_layer2: int,
    hidden_layer3: int,
    hidden_layer4: int,
    hidden_layer5: int,
    hidden_layer6: int,
    hidden_layer7: int,
    hidden_layer8: int,
    hidden_layer9: int,
    hidden_layer10: int,
    dropout_ratio: float,
    lr: float,
) -> List[float]:
    x, y = generate_data(
        N=N, epsilon=epsilon, x_range_from=x_range_from, x_range_to=x_range_to
    )
    loss_list = train(
        x=x,
        y=y,
        save_path=save_path,
        epochs=epochs,
        hidden_layer1=hidden_layer1,
        hidden_layer2=hidden_layer2,
        hidden_layer3=hidden_layer3,
        hidden_layer4=hidden_layer4,
        hidden_layer5=hidden_layer5,
        hidden_layer6=hidden_layer6,
        hidden_layer7=hidden_layer7,
        hidden_layer8=hidden_layer8,
        hidden_layer9=hidden_layer9,
        hidden_layer10=hidden_layer10,
        dropout_ratio=dropout_ratio,
        lr=lr,
    )
    return loss_list
