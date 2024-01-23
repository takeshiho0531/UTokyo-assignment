import numpy as np
import torch
import torch.nn as nn
from DNN import RegressionModel


def inference(model: nn.Module, x_inference: torch.Tensor) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        y_pred = model(x_inference)

    return y_pred.numpy()


def main(
    inf_N: int,
    x_range_from: float,
    x_range_to: float,
    weight_path: str,
    hidden_layer1: int,
    hidden_layer2: int,
) -> np.ndarray:
    model = RegressionModel(hidden_layer1=hidden_layer1, hidden_layer2=hidden_layer2)
    model.load_state_dict(torch.load(weight_path))

    x_inference = (
        torch.FloatTensor(inf_N).uniform_(x_range_from, x_range_to).unsqueeze(1)
    )
    y_pred = inference(model, x_inference)
    return y_pred
