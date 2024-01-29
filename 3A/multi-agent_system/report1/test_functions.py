import numpy as np


def Sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2)


def Rastrigin(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)


def Rosenbrock(x: np.ndarray) -> np.ndarray:
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def Griewank(x: np.ndarray) -> np.ndarray:
    return (
        np.sum(x**2 / 4000)
        - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        + 1
    )


def Alpine(x: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


def two_n_minima(x: np.ndarray) -> np.ndarray:
    return np.sum(x**4 - 16 * x**2 + 5 * x)
