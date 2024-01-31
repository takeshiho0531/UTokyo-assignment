import matplotlib.pyplot as plt
import numpy as np
from pso import pso
from test_functions import Alpine, Griewank, Rastrigin, Rosenbrock, Sphere, two_n_minima


def main(dim: int, sample_num: int, low: float, high: float, step_num: int):
    initial = []

    for i in range(sample_num):
        initial.append(np.random.uniform(low, high, dim))

    test_functions = [Sphere, Rastrigin, Rosenbrock, Griewank, Alpine, two_n_minima]

    for test_function in test_functions:
        print(test_function.__name__)
        result = pso(step_num, initial, criterion=test_function)
        global_best_position_list = []
        for i in range(step_num):
            global_best_position_list.append(result[i]["global_best_position"])
        ideal_position = np.zeros(dim)
        diff_log_list = []
        for i in range(step_num):
            diff_log = np.log(
                np.linalg.norm(ideal_position - global_best_position_list[i])
            )
            diff_log_list.append(diff_log)
        plt.plot(diff_log_list, label=test_function.__name__)
        plt.legend()
        plt.savefig(
            f"3A/multi-agent_system/report1/result/{test_function.__name__}_N_{dim}_{low}_{high}_.png"
        )
        print("--------------------")
