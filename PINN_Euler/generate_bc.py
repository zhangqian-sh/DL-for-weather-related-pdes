import numpy as np
from typing import List, Tuple

def boundary_condition(
    outer_bc_geometry: List[float],
    inner_bc_geometry: List[float],
    bc_num: List[int],
    T_end: float,
):
    """
    Generate BC points for outer and inner boundaries
    """
    x_l, x_r, y_d, y_u = outer_bc_geometry
    xc_l, xc_r, yc_d, yc_u = inner_bc_geometry
    N_x, N_y, N_t, N_bc = bc_num

    N_bc = N_bc // 4 + 1

    # generate bc for outer boundary

    left_points = np.stack((np.ones(N_y) * x_l, np.linspace(y_d, y_u, N_y)), 1)
    right_points = np.stack((np.ones(N_y) * x_r, np.linspace(y_d, y_u, N_y)), 1)
    t_lr = np.repeat(np.linspace(0, T_end, N_t), N_y).reshape(-1, 1)
    X_left = np.hstack((t_lr, np.vstack([left_points for _ in range(N_t)])))
    X_right = np.hstack((t_lr, np.vstack([right_points for _ in range(N_t)])))
    X_lr = np.concatenate((X_left, X_right), 1)
    lr_idx = np.random.choice(len(X_lr), size=N_bc, replace=False)
    X_lr = X_lr[lr_idx]

    down_points = np.stack((np.linspace(x_l, x_r, N_x), np.ones(N_x) * y_d), 1)
    up_points = np.stack((np.linspace(x_l, x_r, N_x), np.ones(N_x) * y_u), 1)
    t_du = np.repeat(np.linspace(0, T_end, N_t), N_x).reshape(-1, 1)
    X_down = np.hstack((t_du, np.vstack([down_points for _ in range(N_t)])))
    X_up = np.hstack((t_du, np.vstack([up_points for _ in range(N_t)])))
    X_du = np.concatenate((X_down, X_up), 1)
    ud_idx = np.random.choice(len(X_du), size=N_bc, replace=False)
    X_du = X_du[ud_idx]

    X_bc_outer = (X_lr, X_du)

    # generate bc for inner boundary

    left_points = np.stack((np.ones(N_y) * xc_l, np.linspace(yc_d, yc_u, N_y)), 1)
    right_points = np.stack((np.ones(N_y) * xc_r, np.linspace(yc_d, yc_u, N_y)), 1)
    t_lr = np.repeat(np.linspace(0, T_end, N_t), N_y).reshape(-1, 1)
    X_left = np.hstack((t_lr, np.vstack([left_points for _ in range(N_t)])))
    X_right = np.hstack((t_lr, np.vstack([right_points for _ in range(N_t)])))
    X_lr = np.concatenate((X_left, X_right), 1)
    lr_idx = np.random.choice(len(X_lr), size=N_bc, replace=False)
    X_lr = X_lr[lr_idx]

    down_points = np.stack((np.linspace(xc_l, xc_r, N_x), np.ones(N_x) * yc_d), 1)
    up_points = np.stack((np.linspace(xc_l, xc_r, N_x), np.ones(N_x) * yc_u), 1)
    t_du = np.repeat(np.linspace(0, T_end, N_t), N_x).reshape(-1, 1)
    X_down = np.hstack((t_du, np.vstack([down_points for _ in range(N_t)])))
    X_up = np.hstack((t_du, np.vstack([up_points for _ in range(N_t)])))
    X_du = np.concatenate((X_down, X_up), 1)
    ud_idx = np.random.choice(len(X_du), size=N_bc, replace=False)
    X_du = X_du[ud_idx]

    X_bc_inner = (X_lr, X_du)

    return X_bc_outer, X_bc_inner
