import numpy as np
import tensorflow as tf
import scipy.io as sio

from PINN_model import PINN_Euler

from typing import Tuple


class InverseTimeDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.step_recorder = 0
        self.lr = 0

    def __call__(self, step):
        self.step_recorder = step
        self.lr = self.initial_learning_rate / (
            1 + self.decay_rate * step / self.decay_steps
        )
        return self.lr


def get_batch_tensor(X: np.ndarray, idx: np.ndarray):
    """
    Convert numpy tensor of shape (N, k) to tensor of shape (N_batch, k) and dtype tf.float32,
    where N = len(X), N_batch = len(idx), and k is the number of input features
    """
    assert len(X.shape) == 2, "Input array is not a PINN input"
    return tf.convert_to_tensor(X[idx], dtype=tf.float32)


def evaluate_and_save(
    model: PINN_Euler,
    X_final: np.ndarray,
    shape: Tuple[int, int],
    loss_history: list,
    save_path: str,
):
    H, W = shape
    t_f, x_f, y_f = X_final[:, 0:1], X_final[:, 1:2], X_final[:, 2:3]
    rho_pred, u_pred, v_pred, p_pred = model.predict_ruvp(t_f, x_f, y_f)
    rho = rho_pred.numpy().reshape(H, W)
    u = u_pred.numpy().reshape(H, W)
    v = v_pred.numpy().reshape(H, W)
    p = p_pred.numpy().reshape(H, W)
    sio.savemat(
        save_path,
        {"rho": rho, "u": u, "v": v, "p": p, "loss": loss_history},
    )
