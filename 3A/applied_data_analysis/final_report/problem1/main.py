import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from inference import inference_deeper_model_wrapper, inference_wrapper
from train import train_deeper_model_wrapper, train_wrapper

seed = 42
torch.manual_seed(seed)


def main(config_file_path: str):
    with open(config_file_path) as file:
        config = yaml.safe_load(file)

    # train
    if config["deeper_model"]:
        loss_list = train_deeper_model_wrapper(
            N=config["N"],
            epsilon=config["epsilon"],
            x_range_from=config["x_range_from"],
            x_range_to=config["x_range_to"],
            save_path=config["weight_path"],
            epochs=config["epochs"],
            hidden_layer1=config["hidden_layer1"],
            hidden_layer2=config["hidden_layer2"],
            hidden_layer3=config["hidden_layer3"],
            hidden_layer4=config["hidden_layer4"],
            hidden_layer5=config["hidden_layer5"],
            hidden_layer6=config["hidden_layer6"],
            hidden_layer7=config["hidden_layer7"],
            hidden_layer8=config["hidden_layer8"],
            hidden_layer9=config["hidden_layer9"],
            hidden_layer10=config["hidden_layer10"],
            dropout_ratio=config["dropout_ratio"],
            lr=config["lr"],
        )
    else:
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
    if config["deeper_model"]:
        x_inference_sorted, y_pred = inference_deeper_model_wrapper(
            inf_N=config["inf_N"],
            x_range_from=config["x_range_from"],
            x_range_to=config["x_range_to"],
            weight_path=config["weight_path"],
            hidden_layer1=config["hidden_layer1"],
            hidden_layer2=config["hidden_layer2"],
            hidden_layer3=config["hidden_layer3"],
            hidden_layer4=config["hidden_layer4"],
            hidden_layer5=config["hidden_layer5"],
            hidden_layer6=config["hidden_layer6"],
            hidden_layer7=config["hidden_layer7"],
            hidden_layer8=config["hidden_layer8"],
            hidden_layer9=config["hidden_layer9"],
            hidden_layer10=config["hidden_layer10"],
            dropout_ratio=config["dropout_ratio"],
        )
    else:
        x_inference_sorted, y_pred = inference_wrapper(
            inf_N=config["inf_N"],
            x_range_from=config["x_range_from"],
            x_range_to=config["x_range_to"],
            weight_path=config["weight_path"],
            hidden_layer1=config["hidden_layer1"],
            hidden_layer2=config["hidden_layer2"],
            dropout_ratio=config["dropout_ratio"],
        )

    def generate_true_data(x: torch.Tensor, epsilon: float) -> torch.Tensor:
        y = 5 * x * torch.sin(2 * np.pi * x) + 4 * torch.exp(1 / (x + 1)) + epsilon
        return y

    y_true = generate_true_data(x_inference_sorted, epsilon=config["epsilon"])

    # inference graph
    plt.figure(figsize=(8, 6))
    plt.plot(
        x_inference_sorted.numpy(),
        y_pred,
        label="predicted",
        marker="o",
        linestyle="-",
        color="r",
    )
    plt.plot(
        x_inference_sorted.numpy(),
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
    plt.savefig(config["inference_result_graph_path"])


if __name__ == "__main__":
    main(sys.argv[1])
