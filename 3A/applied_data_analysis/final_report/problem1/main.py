import matplotlib.pyplot as plt
import yaml
from inference import inference_wrapper
from train import train_wrapper

with open("config.yaml") as file:
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

# loss のグラフ出力

# inference
y_pred = inference_wrapper(
    inf_N=config["inf_N"],
    x_range_from=config["x_range_from"],
    x_range_to=config["x_range_to"],
    weight_path=config["weight_path"],
    hidden_layer1=config["hidden_layer1"],
    hidden_layer2=config["hidden_layer2"],
)

# plot
