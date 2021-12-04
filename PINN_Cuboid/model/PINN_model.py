import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.optimizers import Adam
import scipy.io as sio

from typing import List, Tuple

from utils import InverseTimeDecay


class PINNModel(tf.keras.Model):
    # initialize the model
    def __init__(self, layers: List[int]) -> None:
        super(PINNModel, self).__init__()
        # set up network
        self.layers_list = layers
        self.net = self.create_network(layers)

    def create_network(self, layers: List[int]):
        hidden_layers = [
            Dense(
                layer,
                activation="tanh",  # Lambda(lambda x: tf.sin(x)),
                kernel_initializer="glorot_normal",
            )
            for layer in layers[1:-1]
        ]
        output_layer = [Dense(layers[-1], kernel_initializer="glorot_normal")]
        return hidden_layers + output_layer

    def dummy_model_for_summary(self):
        x = Input(shape=(self.layers_list[0]))
        out = x
        for layer in self.net:
            out = layer(out)
        return tf.keras.Model(inputs=[x], outputs=out)

    # forward pass
    @tf.function
    def call(self, t, x, y):
        X = tf.concat([t, x, y], 1)
        for layer in self.net:
            X = layer(X)
        return X