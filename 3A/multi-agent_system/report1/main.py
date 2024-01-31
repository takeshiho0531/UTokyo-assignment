import matplotlib.pyplot as plt
import numpy as np
import yaml
from pso import pso
from test_functions import Alpine, Griewank, Rastrigin, Rosenbrock, Sphere, two_n_minima


def main(config_file_path: str):
    with open(config_file_path) as file:
        config = yaml.safe_load(file)

    dim = config["dim"]
    sample_num = config["sample_num"]
    low = config["low"]
    high = config["high"]
    step_num = config["step_num"]

    initial = []
    for i in range(sample_num):
        initial.append(np.random.uniform(low, high, dim))

    test_function = config["test_function"]
    if test_function == "Sphere":
        test_function = Sphere
    elif test_function == "Rastrigin":
        test_function = Rastrigin
    elif test_function == "Rosenbrock":
        test_function = Rosenbrock
    elif test_function == "Griewank":
        test_function = Griewank
    elif test_function == "Alpine":
        test_function = Alpine
    elif test_function == "two_n_minima":
        test_function = two_n_minima
    else:
        raise ValueError("test_function is invalid")

    ideal_position = np.zeros(dim)

    print(test_function.__name__)
    result = pso(step_num, initial, criterion=test_function)
    global_best_position_list = []
    diff_log_list = []
    for i in range(step_num):
        global_best_position_list.append(result[i]["global_best_position"])
        diff_log = np.log(np.linalg.norm(ideal_position - global_best_position_list[i]))
        diff_log_list.append(diff_log)
        print(f"step {i}: {diff_log}")
    plt.plot(diff_log_list, label=test_function.__name__)
    plt.xlabel("step")
    plt.ylabel("log(diff)")
    plt.title(
        f"dim={dim}, sample_num={sample_num}, low={low}, high={high}, test_function={test_function.__name__}"
    )
    plt.legend()
    plt.savefig(
        f"3A/multi-agent_system/report1/result/{test_function.__name__}_N_{dim}_{low}_{high}.png"
    )
    print("--------------------")
