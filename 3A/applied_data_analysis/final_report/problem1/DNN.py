import torch.nn as nn
import torch.nn.functional as F


class RegressionModel(nn.Module):
    def __init__(self, hidden_layer1, hidden_layer2, dropout_ratio):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, hidden_layer1)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc3 = nn.Linear(hidden_layer2, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
