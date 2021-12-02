# %% import modules
from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam

import scipy.io as sio
import time
import os

from PINN_model import PINN_Euler, evaluate_and_save
from generate_bc import boundary_condition
from utils import InverseTimeDecay, get_batch_tensor

# %% set parameters
gamma = 1.4
rho_0 = 1.225
u_0, v_0 = 100, 0  # 100, 0
p_0 = 1.013 * 10
u_boundary = 100

scale = (1, 100, 100, 10)

# geometry
x_l, x_r = 0, 2
y_d, y_u = 0, 2
xc_l, xc_r = 0.75, 1.25
yc_d, yc_u = 0.75, 1.25

T = 1
N_x, N_y, N_t = 100, 100, 100

N_res = 100000
N_ic = 1000
N_bc = 10000
batch_size = 100000
num_epochs = 20000

decay_step_scale = 5000

num_layer = 4
num_node = 100
layers = [3] + num_layer * [num_node] + [4]

lambda_eqn, lambda_ic, lambda_bc = 10, 100, 100

job_name = "cuboid"
save_path = f"../Results/PINN_Euler/{job_name}/"
os.makedirs(save_path, exist_ok=True)


def intialize(x: np.ndarray, y: np.ndarray):
    L_x, L_y = x.shape
    rho_grid, u_grid, v_grid, p_grid = (
        np.ones_like(x),
        np.ones_like(x),
        np.ones_like(x),
        np.ones_like(x),
    )
    rho_grid = rho_grid * rho_0
    u_grid = u_grid * u_0
    v_grid = v_grid * v_0
    p_grid = p_grid * p_0

    mask_grid = np.ones_like(x)
    mask_grid[(x > xc_l) & (x < xc_r) & (y > yc_d) & (y < yc_u)] = 0
    return rho_grid, u_grid, v_grid, p_grid, mask_grid


# %% generate sample points
# Initial Condition
x, y = np.linspace(x_l, x_r, N_x), np.linspace(y_d, y_u, N_y)
x_grid, y_grid = np.meshgrid(x, y)
x, y = x_grid.flatten(), y_grid.flatten()
t = np.zeros_like(x)
rho_grid, u_grid, v_grid, p_grid, mask = intialize(x_grid, y_grid)
rho, u, v, p = rho_grid[mask > 0], u_grid[mask > 0], v_grid[mask > 0], p_grid[mask > 0]
x, y = x_grid[mask > 0], y_grid[mask > 0]
t = np.zeros_like(x)
X_ic = np.stack((t, x, y, rho, u, v, p), 1)
del t, x, y, rho, u, v, p
ic_idx = np.random.choice(len(X_ic), size=N_ic, replace=False)
X_ic = X_ic[ic_idx]

# PDE residual
x = np.random.uniform(x_l, x_r, N_res * 2)
y = np.random.uniform(y_d, y_u, N_res * 2)
t = np.random.uniform(0, T, N_res * 2)
perm = lambda x: np.random.permutation(x)
X_res = np.stack((perm(t), perm(x), perm(y)), 1)
del t, x, y
print(X_res.shape)


def in_computation_area(point: Tuple[float, float]):
    x, y = point
    if x > xc_l and x < xc_r and y > yc_d and y < yc_u:
        return False
    else:
        return True


in_area_idx = list(map(in_computation_area, X_res[:, 1:3]))
X_res = X_res[in_area_idx]
print(X_res.shape)
res_idx = np.random.choice(len(X_res), size=N_res, replace=False)
X_res = X_res[res_idx]
print(X_res.shape)

# Boundary Condition
outer_bc_geometry = [x_l, x_r, y_d, y_u]
inner_bc_geometry = [xc_l, xc_r, yc_d, yc_u]
bc_num = N_x, N_y, N_t, N_bc
X_bc_outer, X_bc_inner = boundary_condition(
    outer_bc_geometry, inner_bc_geometry, bc_num, T
)
print([_.shape for _ in X_bc_inner])
mask = np.ones((N_y, N_x))
x_grid, y_grid = np.meshgrid(np.linspace(x_l, x_r, N_x), np.linspace(y_d, y_u, N_y))
mask[(x_grid > xc_l) & (x_grid < xc_r) & (y_grid > yc_d) & (y_grid < yc_u)] = np.nan

# Final points
x, y = np.linspace(x_l, x_r, N_x), np.linspace(y_d, y_u, N_y)
x_grid, y_grid = np.meshgrid(x, y)
x, y = x_grid.flatten(), y_grid.flatten()
t = np.ones_like(x) * T
X_final = np.stack((t, x, y), 1)


# %% define model and optimizer
decay_step = decay_step_scale * N_res // batch_size

model = PINN_Euler(
    layers, gamma, scale, u_boundary, (lambda_eqn, lambda_ic, lambda_bc), decay_step
)
model.dummy_model_for_summary().summary()

# %% train for an epoch
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
    batch_num = len(res_batch_idx_list)
    # batch data for initial condition
    batch_size_ic = N_ic // batch_num + 1
    ic_batch_idx_list = mini_batch(N_ic, batch_size_ic)
    # batch data for boundary condition
    X_bc_outer, X_bc_inner = X_bc
    X_lr, X_du = X_bc_outer
    Xc_lr, Xc_du = X_bc_inner
    batch_size_bc = len(X_lr) // batch_num + 1
    bc_batch_idx_list = mini_batch(len(X_lr), batch_size_bc)
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
        Xc_lr_batch = get_batch_tensor(Xc_lr, bc_idx)
        Xc_du_batch = get_batch_tensor(Xc_du, bc_idx)
        X_bc_batch = (X_lr_batch, X_du_batch, Xc_lr_batch, Xc_du_batch)
        # train for a batch
        loss_list = model.train_step(X_res_batch, X_ic_batch, X_bc_batch)
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
    ) = train(model, X_res, X_ic, (X_bc_outer, X_bc_inner))
    loss_Euler = (
        loss_eqn_mass
        + loss_eqn_momentum_x
        + loss_eqn_momentum_y
        + loss_eqn_energy
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
        evaluate_and_save(model, X_final, mask, [N_y, N_x], log_loss, save_file_path)
