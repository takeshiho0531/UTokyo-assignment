import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from inference import inference_wrapper
from train import train_wrapper


def main(config_file_path: str):
    with open(config_file_path) as file:
        config = yaml.safe_load(file)

    # train
    loss_list = train_wrapper(
        N=config["N"],
        epsilon=config["epsilon"],
        x_range_from=config["x_range_from"],
        x_range_to=config["x_range_to"],
        save_path=config["weight_path"],
        epochs=config["epochs"],
        hidden_layer1=config["hidden_layer1"],
        hidden_layer2=config["hidden_layer2"],
        dropout_ratio=config["dropout_ratio"],
        lr=config["lr"],
    )

    # loss graph
    plt.plot(loss_list)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title("Changes in Loss")
    plt.grid(True)
    plt.savefig(config["loss_graph_path"])

    # inference
    x_inference, y_pred = inference_wrapper(
        inf_N=config["inf_N"],
        x_range_from=config["x_range_from"],
        x_range_to=config["x_range_to"],
        weight_path=config["weight_path"],
        hidden_layer1=config["hidden_layer1"],
        hidden_layer2=config["hidden_layer2"],
    )

    def generate_true_data(x: torch.Tensor, epsilon: float) -> torch.Tensor:
        y = 5 * x * torch.sin(2 * np.pi * x) + 4 * torch.exp(1 / (x + 1)) + epsilon
        return y

    y_true = generate_true_data(x_inference, epsilon=config["epsilon"])

    # inference graph
    plt.figure(figsize=(8, 6))
    plt.plot(
        x_inference.numpy(),
        y_pred,
        label="predicted",
        marker="o",
        linestyle="-",
        color="r",
    )
    plt.plot(
        x_inference.numpy(),
        y_true.numpy(),
        label="true",
        marker="s",
        linestyle="--",
        color="b",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Overlayed Data (predicted & true)")
    plt.legend()
    plt.grid(True)
    plt.show(config["inference_result_graph_path"])


if __name__ == "__main__":
    main(sys.argv[1])
