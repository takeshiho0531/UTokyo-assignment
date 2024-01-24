import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DNN import RegressionDeeperModel, RegressionModel
from torch.utils.data import DataLoader, TensorDataset


def generate_data(N, epsilon, x_range_from=0, x_range_to=1):
    x = torch.FloatTensor(N).uniform_(x_range_from, x_range_to).unsqueeze(1)
    y = 5 * x * torch.sin(2 * np.pi * x) + 4 * torch.exp(1 / (x + 1)) + epsilon
    return x, y


def train(
    x, y, val_split, save_path, epochs, hidden_layer1, hidden_layer2, dropout_ratio, lr
):
    total_size = len(x)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:], y[train_size:]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_loss_list = []
    val_loss_list = []

    model = RegressionModel(
        hidden_layer1=hidden_layer1,
        hidden_layer2=hidden_layer2,
        dropout_ratio=dropout_ratio,
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    return train_loss_list, val_loss_list


def train_deeper_model(
    x,
    y,
    val_split,
    save_path,
    epochs,
    hidden_layer1,
    hidden_layer2,
    hidden_layer3,
    hidden_layer4,
    hidden_layer5,
    hidden_layer6,
    hidden_layer7,
    hidden_layer8,
    hidden_layer9,
    hidden_layer10,
    dropout_ratio,
    lr,
):
    total_size = len(x)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:], y[train_size:]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_loss_list = []
    val_loss_list = []

    model = RegressionDeeperModel(
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
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    return train_loss_list, val_loss_list


def train_wrapper(
    N: int,
    epsilon: float,
    x_range_from: float,
    x_range_to: float,
    val_split: float,
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
    train_loss_list, val_loss_list = train(
        x=x,
        y=y,
        val_split=val_split,
        save_path=save_path,
        epochs=epochs,
        hidden_layer1=hidden_layer1,
        hidden_layer2=hidden_layer2,
        dropout_ratio=dropout_ratio,
        lr=lr,
    )
    return train_loss_list, val_loss_list


def train_deeper_model_wrapper(
    N: int,
    epsilon: float,
    x_range_from: float,
    x_range_to: float,
    val_split: float,
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
    train_loss_list, val_loss_list = train_deeper_model(
        x=x,
        y=y,
        val_split=val_split,
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
    return train_loss_list, val_loss_list
