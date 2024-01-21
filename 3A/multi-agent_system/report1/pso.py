import random
import numpy as np


def update_position(x, y, vx, vy):
    updated_x = x + vx
    updated_y = y + vy
    return updated_x, updated_y


def update_velosity(x, y, vx, vy, w, c1, c2, xp, xg, yp, yg):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    updated_vx = w * vx + c1 * r1(xp - x) + c2 * r2 * (xg - x)
    updated_vy = w * vy + c1 * r1(yp - y) + c2 * r2 * (yg - y)
    return updated_vx, updated_vy

def main():
    N = 100
    x_min, x_max = -5,5
    y_min, y_max = -5,5
    #粒子位置, 速度, パーソナルベスト, グローバルベストの初期化を行う
    ps = [{"x": random.uniform(x_min, x_max),
           "y": random.uniform(y_min, y_max)} for i in range(N)]
    vs = [{"x": 0.0, "y": 0.0} for i in range(N)]
    personal_best_positions = list(ps)
    personal_best_scores = [criterion(p["x"], p["y"]) for p in ps]
    best_particle = np.argmin(personal_best_positions)
    global_best_position = personal_best_positions[best_particle]