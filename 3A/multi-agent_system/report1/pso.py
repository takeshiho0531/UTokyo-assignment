import random

import numpy as np


def update_position(x, y, vx, vy):
    updated_x = x + vx
    updated_y = y + vy
    return updated_x, updated_y


def update_velosity(x, y, vx, vy, xp, xg, yp, yg, w=0.5, c1=0.14, c2=0.14):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    updated_vx = w * vx + c1 * r1(xp - x) + c2 * r2 * (xg - x)
    updated_vy = w * vy + c1 * r1(yp - y) + c2 * r2 * (yg - y)
    return updated_vx, updated_vy


def main():
    N = 100
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    # 粒子位置, 速度, パーソナルベスト, グローバルベストの初期化を行う
    ps = [
        {"x": random.uniform(x_min, x_max), "y": random.uniform(y_min, y_max)}
        for i in range(N)
    ]
    vs = [{"x": 0.0, "y": 0.0} for i in range(N)]
    personal_best_positions = list(ps)
    personal_best_scores = [criterion(p["x"], p["y"]) for p in ps]
    best_particle = np.argmin(personal_best_positions)
    global_best_position = personal_best_positions[best_particle]

    T = 30
    for t in range(T):
        for n in range(N):
            x, y = ps[n]["x"], ps[n]["y"]
            vx, vy = vs[n]["x"], vs[n]["y"]
            p = personal_best_positions[n]

            updated_x, updated_y = update_position(x, y, vx, vy)
            ps[n] = {"x": updated_x, "y": updated_y}

            xg = global_best_position["x"]
            yg = global_best_position["y"]
            xp = personal_best_positions["x"]
            yp = personal_best_positions["y"]
            updated_vx, updated_vy = update_velosity(x, y, vx, vy, xp, xg, yp, yg)
            vs = {"x": updated_vx, "y": updated_vy}

            score = criterion(updated_x, updated_y)
            if score < personal_best_scores[n]:
                personal_best_scores[n] = score
                personal_best_positions[n] = {"x": updated_x, "y": updated_y}

        best_particle = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[best_particle]

        print(global_best_position)
        print(min(personal_best_scores))


if __name__ == "__main__":
    main()
