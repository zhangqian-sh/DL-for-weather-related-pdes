import numpy as np
import tensorflow as tf


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
