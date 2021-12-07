# %% import modules
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam

import scipy.io as sio
import time
import os

from PINN_model import PINN_Euler
from utils import InverseTimeDecay, get_batch_tensor, evaluate_and_save

# %% set parameters
rho_l, rho_r = 1, 1
u_l, u_r = -2, 2
v_l, v_r = 0, 0
p_l, p_r = 0.4, 0.4
gamma = 1.4

# example 1
# x_l, x_r = 0, 2 * np.pi
# y_d, y_u = 0, 2 * np.pi
# T = 1
# N_x, N_y, N_t = 200, 200, 100

# example 2
x_l, x_r = 0, 10
y_d, y_u = 0, 10
x_c, y_c = 5, 5
T = 1
epsilon = 5 / (2 * np.pi)
alpha = 1 / 2
N_x, N_y, N_t = 1000, 1000, 100

# example 4
# x_l, x_r = -1, 1
# y_d, y_u = -1, 1
# T = 0.15
# N_x, N_y, N_t = 200, 200, 150

N_res = 100000
N_ic = 1000
N_bc = 10000
batch_size = 100000
num_epochs = 20000

decay_step_scale = 2500

num_layer = 4
num_node = 100
layers = [3] + num_layer * [num_node] + [4]

lambda_eqn, lambda_ic, lambda_bc = 1, 100, 1

job_name = "example_2"
save_path = f"../Results/PINN_Euler/{job_name}/"
os.makedirs(save_path, exist_ok=True)


# example 2 exact solution at t=0
def initial_exact_solution(x, y):
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    phi = epsilon * np.exp(alpha * (1 - r ** 2))
    Txy = 1 - (gamma - 1) / (2 * gamma) * phi ** 2
    rho = Txy ** (1 / (gamma - 1))
    u = 1 - (y - y_c) * phi
    v = 1 + (x - x_c) * phi
    p = Txy ** (gamma / (gamma - 1))
    return rho, u, v, p


def intialize(x: np.ndarray, y: np.ndarray):
    L_y, L_x = x.shape
    rho_grid, u_grid, v_grid, p_grid = (
        np.zeros_like(x),
        np.zeros_like(x),
        np.zeros_like(x),
        np.zeros_like(x),
    )
    # ex1
    # rho_grid = 1 + 0.5 * np.sin(x + y)
    # ex4
    # rho_grid[x < 0], rho_grid[x >= 0] = rho_l, rho_r
    # u_grid[x < 0], u_grid[x >= 0] = u_l, u_r
    # v_grid[x < 0], v_grid[x >= 0] = v_l, v_r
    # p_grid[x < 0], p_grid[x >= 0] = p_l, p_r
    # ex2
    # for i in range(L_y):
    #     for j in range(L_x):
    #         (
    #             rho_grid[i, j],
    #             u_grid[i, j],
    #             v_grid[i, j],
    #             p_grid[i, j],
    #         ) = initial_exact_solution(x[i, j], y[i, j])
    rho_grid, u_grid, v_grid, p_grid = initial_exact_solution(x, y)

    return rho_grid, u_grid, v_grid, p_grid


# %% generate sample points
# PDE residual
x = np.random.uniform(x_l, x_r, N_res)
y = np.random.uniform(y_d, y_u, N_res)
t = np.random.uniform(0, T, N_res)
perm = lambda x: np.random.permutation(x)
X_res = np.stack((perm(t), perm(x), perm(y)), 1)
del t, x, y
res_idx = np.random.choice(len(X_res), size=N_res, replace=False)
X_res = X_res[res_idx]

# Initial Condition
x, y = np.linspace(x_l, x_r, N_x), np.linspace(y_d, y_u, N_y)
x_grid, y_grid = np.meshgrid(x, y)
x, y = x_grid.flatten(), y_grid.flatten()
t = np.zeros_like(x)
rho_grid, u_grid, v_grid, p_grid = intialize(x_grid, y_grid)
rho, u, v, p = rho_grid.flatten(), u_grid.flatten(), v_grid.flatten(), p_grid.flatten()
X_ic = np.stack((t, x, y, rho, u, v, p), 1)
del t, x, y
ic_idx = np.random.choice(len(X_ic), size=N_ic, replace=False)
X_ic = X_ic[ic_idx]

# Boundary Condition
def boundary_value(X: np.ndarray):
    t, x, y = X.T
    rho, u, v, p = (
        np.zeros_like(t),
        np.zeros_like(t),
        np.zeros_like(t),
        np.zeros_like(t),
    )
    # for i in range(len(t)):
    #     rho[i], u[i], v[i], p[i] = initial_exact_solution(x[i] - t[i], y[i] - t[i])
    rho, u, v, p = initial_exact_solution(x - t, y - t)
    Y = np.stack((rho, u, v, p), 1)
    return Y


left_points = np.stack((np.ones(N_y) * x_l, np.linspace(y_d, y_u, N_y)), 1)
right_points = np.stack((np.ones(N_y) * x_r, np.linspace(y_d, y_u, N_y)), 1)
t_lr = np.repeat(np.linspace(0, T, N_t), N_y).reshape(-1, 1)
X_left = np.hstack((t_lr, np.vstack([left_points for _ in range(N_t)])))
Y_left = boundary_value(X_left)
X_right = np.hstack((t_lr, np.vstack([right_points for _ in range(N_t)])))
Y_right = boundary_value(X_right)
X_lr = np.concatenate((X_left, Y_left, X_right, Y_right), 1)
lr_idx = np.random.choice(len(X_lr), size=N_bc, replace=False)
X_lr = X_lr[lr_idx]

down_points = np.stack((np.linspace(x_l, x_r, N_x), np.ones(N_x) * y_d), 1)
up_points = np.stack((np.linspace(x_l, x_r, N_x), np.ones(N_x) * y_u), 1)
t_du = np.repeat(np.linspace(0, T, N_t), N_x).reshape(-1, 1)
X_down = np.hstack((t_du, np.vstack([down_points for _ in range(N_t)])))
Y_down = boundary_value(X_down)
X_up = np.hstack((t_du, np.vstack([up_points for _ in range(N_t)])))
Y_up = boundary_value(X_up)
X_du = np.concatenate((X_down, Y_down, X_up, Y_up), 1)
ud_idx = np.random.choice(len(X_du), size=N_bc, replace=False)
X_du = X_du[ud_idx]

X_bc = (X_lr, X_du)

# Final points
x, y = np.linspace(x_l, x_r, N_x), np.linspace(y_d, y_u, N_y)
x_grid, y_grid = np.meshgrid(x, y)
x, y = x_grid.flatten(), y_grid.flatten()
t = np.ones_like(x) * T
X_final = np.stack((t, x, y), 1)


# %% define model and optimizer
model = PINN_Euler(layers, gamma, (lambda_eqn, lambda_ic, lambda_bc))
model.dummy_model_for_summary().summary()

decay_step = decay_step_scale * N_res // batch_size
optimizer = Adam(learning_rate=InverseTimeDecay(0.001, decay_step, 0.5))

# %% train for an epoch
# train for an epoch
def mini_batch(N: int, batch_size: int):
    return np.split(
        np.random.permutation(N),
        np.arange(batch_size, N, batch_size),
    )


def train(model: PINN_Euler, X_res, X_ic, X_bc):
    """
    loss_avg shape
    [loss, loss_supervised, loss_eqn_momentum_x, loss_eqn_momentum_y, loss_eqn_mass, loss_bc]
    """
    N_residual = len(X_res)
    N_ic = len(X_ic)
    # batch data for residue constraint
    res_batch_idx_list = mini_batch(N_residual, batch_size)
    # batch data for initial condition
    batch_num = len(res_batch_idx_list)
    batch_size_ic = N_ic // batch_num + 1
    ic_batch_idx_list = mini_batch(N_ic, batch_size_ic)
    # batch data for boundary condition
    X_lr, X_du = X_bc
    batch_size_bc = N_bc // batch_num + 1
    bc_batch_idx_list = mini_batch(N_bc, batch_size_bc)
    # loop through batches
    loss_array = []
    for ((_, res_idx), (_, ic_idx), (_, bc_idx)) in zip(
        enumerate(res_batch_idx_list),
        enumerate(ic_batch_idx_list),
        enumerate(bc_batch_idx_list),
    ):
        # gather and convert data
        X_res_batch = get_batch_tensor(X_res, res_idx)
        X_ic_batch = get_batch_tensor(X_ic, ic_idx)
        X_lr_batch = get_batch_tensor(X_lr, bc_idx)
        X_du_batch = get_batch_tensor(X_du, bc_idx)
        # train for a batch
        loss_list = model.train_step(X_res_batch, X_ic_batch, (X_lr_batch, X_du_batch))
    loss_array.append(loss_list)
    loss_avg = np.mean(np.array(loss_array), axis=0)

    return loss_avg


# log
log_loss = []

# %% training loop
for epoch in range(1, num_epochs + 1):
    # train for a single epoch
    start_time = time.time()
    (
        loss,
        loss_eqn_mass,
        loss_eqn_momentum_x,
        loss_eqn_momentum_y,
        loss_eqn_energy,
        loss_ic,
        loss_bc,
    ) = train(model, X_res, X_ic, X_bc)
    loss_Euler = (
        loss_eqn_mass + loss_eqn_momentum_x + loss_eqn_momentum_y + loss_eqn_energy
    )
    end_time = time.time()

    # visualize and log loss
    optim_step = epoch * (N_res // batch_size)
    lr = 0.001 / (1 + 0.5 * optim_step / decay_step)
    print("-" * 50)
    print(
        f"Epoch: {epoch:d}, It: {optim_step:d}, Time: {end_time-start_time:.2f}s, Learning Rate: {lr:.1e}"
    )
    print(
        f"Epoch: {epoch:d}, It: {optim_step:d}, Loss_sum: {loss:.3e}, Loss_Euler: {loss_Euler:.3e}, Loss_ic: {loss_ic:.3e}, Loss_bc: {loss_bc:.3e}"
    )
    print(
        f"Epoch: {epoch:d}, It: {optim_step:d}, Loss_e_m: {loss_eqn_mass:.3e}, Loss_e_x: {loss_eqn_momentum_x:.3e}, Loss_e_y: {loss_eqn_momentum_y:.3e},Loss_e_E: {loss_eqn_energy:.3e}"
    )

    log_loss.append(
        [
            loss,
            loss_eqn_mass,
            loss_eqn_momentum_x,
            loss_eqn_momentum_y,
            loss_eqn_energy,
            loss_ic,
            loss_bc,
        ]
    )

    if epoch % 1000 == 0:
        save_file_path = save_path + f"PINN_results_{epoch}.mat"
        evaluate_and_save(model, X_final, [N_y, N_x], log_loss, save_file_path)
