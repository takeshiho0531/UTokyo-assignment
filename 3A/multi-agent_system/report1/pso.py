import random
from typing import List

import numpy as np


def update_position(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    updated_x = x + v
    return updated_x


def update_velosity(
    x: np.ndarray,
    v: np.ndarray,
    personal_best_position: np.ndarray,
    global_best_position: np.ndarray,
    w=0.5,
    c1=0.14,
    c2=0.14,
) -> np.ndarray:
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    updated_v = (
        w * v
        + c1 * r1(personal_best_position - x)
        + c2 * r2 * (global_best_position - x)
    )
    return updated_v


def main(
    num: int,
    step_num: int,
    initial: List[np.ndarray],  # 各要素にその点の全次元位置情報が含まれる
    criterion,
):
    # initialize
    assert len(initial) == num
    position_list = initial
    velosity_list = np.zeros(initial.shape)
    personal_best_position_list = list(position_list)
    personal_best_score_list = [criterion(position_list) for p in position_list]
    best_particle_idx = np.argmin(personal_best_score_list)
    global_best_position = personal_best_position_list[best_particle_idx]

    for t in range(step_num):
        for i in range(num):
            x = position_list[i]
            v = velosity_list[i]
            personal_best_position = personal_best_position_list[i]

            updated_x = update_position(x, v)
            position_list[i] = updated_x

            updated_v = update_velosity(
                x, v, personal_best_position, global_best_position
            )
            velosity_list[i] = updated_v

            score = criterion(updated_x)
            if score < personal_best_score_list[i]:
                personal_best_score_list[i] = score
                personal_best_position_list[i] = updated_x

        best_particle_idx = np.argmin(personal_best_score_list)
        global_best_position = personal_best_position_list[best_particle_idx]

        print(global_best_position)
        print(min(personal_best_score_list))


if __name__ == "__main__":
    main()
