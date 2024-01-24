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


class RegressionDeeperModel(nn.Module):
    def __init__(
        self,
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
    ):
        super(RegressionDeeperModel, self).__init__()
        self.fc1 = nn.Linear(1, hidden_layer1)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.dropout3 = nn.Dropout(dropout_ratio)
        self.fc4 = nn.Linear(hidden_layer3, hidden_layer4)
        self.dropout4 = nn.Dropout(dropout_ratio)
        self.fc5 = nn.Linear(hidden_layer4, hidden_layer5)
        self.dropout5 = nn.Dropout(dropout_ratio)
        self.fc6 = nn.Linear(hidden_layer5, hidden_layer6)
        self.dropout6 = nn.Dropout(dropout_ratio)
        self.fc7 = nn.Linear(hidden_layer6, hidden_layer7)
        self.dropout7 = nn.Dropout(dropout_ratio)
        self.fc8 = nn.Linear(hidden_layer7, hidden_layer8)
        self.dropout8 = nn.Dropout(dropout_ratio)
        self.fc9 = nn.Linear(hidden_layer8, hidden_layer9)
        self.dropout9 = nn.Dropout(dropout_ratio)
        self.fc10 = nn.Linear(hidden_layer9, hidden_layer10)
        self.dropout10 = nn.Dropout(dropout_ratio)
        self.fc11 = nn.Linear(hidden_layer10, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        x = self.dropout4(F.relu(self.fc4(x)))
        x = self.dropout5(F.relu(self.fc5(x)))
        x = self.dropout6(F.relu(self.fc6(x)))
        x = self.dropout7(F.relu(self.fc7(x)))
        x = self.dropout8(F.relu(self.fc8(x)))
        x = self.dropout9(F.relu(self.fc9(x)))
        x = self.dropout10(F.relu(self.fc10(x)))
        x = self.fc11(x)
        return x
