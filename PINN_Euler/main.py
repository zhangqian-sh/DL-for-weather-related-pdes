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

T = 0.15
x_l, x_r = -1, 1
y_d, y_u = -1, 1
N_x, N_y, N_t = 200, 100, 150

N_res = 50000
N_ic = 1000
batch_size = 10000
num_epochs = 2

lambda_eqn, lambda_ic = 1, 100

job_name = "test"
save_path = f"../Results/PINN_Euler/{job_name}/"
os.makedirs(save_path, exist_ok=True)


def intialize(x: np.ndarray, y: np.ndarray):
    L_x, L_y = x.shape
    rho_grid, u_grid, v_grid, p_grid = (
        np.zeros_like(x),
        np.zeros_like(x),
        np.zeros_like(x),
        np.zeros_like(x),
    )
    rho_grid[x < 0], rho_grid[x >= 0] = rho_l, rho_r
    u_grid[x < 0], u_grid[x >= 0] = u_l, u_r
    v_grid[x < 0], v_grid[x >= 0] = v_l, v_r
    p_grid[x < 0], p_grid[x >= 0] = p_l, p_r
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
X_ic = np.random.permutation(X_ic)

# Final points
x, y = np.linspace(x_l, x_r, N_x), np.linspace(y_d, y_u, N_y)
x_grid, y_grid = np.meshgrid(x, y)
x, y = x_grid.flatten(), y_grid.flatten()
t = np.ones_like(x) * T
X_final = np.stack((t, x, y), 1)


# %% define model and optimizer
num_layer = 4
num_node = 100
layers = [3] + num_layer * [num_node] + [4]
model = PINN_Euler(layers, gamma, (lambda_eqn, lambda_ic))
model.dummy_model_for_summary().summary()

decay_step = 500 * N_res // batch_size
optimizer = Adam()  # Adam(learning_rate=InverseTimeDecay(0.001, decay_step, 0.5))

# %% train for an epoch
# train for an epoch
def mini_batch(N: int, batch_size: int):
    return np.split(
        np.random.permutation(N),
        np.arange(batch_size, N, batch_size),
    )


def train(model: PINN_Euler, X_res, X_ic):
    """
    loss_avg shape
    [loss, loss_supervised, loss_eqn_momentum_x, loss_eqn_momentum_y, loss_eqn_mass, loss_bc]
    """
    N_residual = len(X_res)
    N_ic = len(X_ic)
    # batch data for residue constraint
    res_batch_idx_list = mini_batch(N_residual, batch_size)
    # batch data for boundary condition
    batch_num = len(res_batch_idx_list)
    batch_size_ic = N_ic // batch_num + 1
    ic_batch_idx_list = mini_batch(N_ic, batch_size_ic)
    # loop through batches
    loss_array = []
    for ((res_batch_idx, res_idx), (ic_batch_idx, ic_idx),) in zip(
        enumerate(res_batch_idx_list),
        enumerate(ic_batch_idx_list),
    ):
        # gather and convert data
        X_res_batch = get_batch_tensor(X_res, res_idx)
        X_ic_batch = get_batch_tensor(X_ic, ic_idx)
        # train for a batch
        loss_list = model.train_step(X_res_batch, X_ic_batch)
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
    ) = train(model, X_res, X_ic)
    # print(model.trainable_variables)
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
        f"Epoch: {epoch:d}, It: {optim_step:d}, Loss_sum: {loss:.3e}, Loss_Euler: {loss_Euler:.3e}, Loss_ic: {loss_ic:.3e}, gamma: {model.gamma:.3e}"
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
        ]
    )

    if epoch % 500 == 0:
        save_file_path = save_path + f"PINN_results_{epoch}.mat"
        evaluate_and_save(model, X_final, [N_y, N_x], log_loss, save_file_path)
