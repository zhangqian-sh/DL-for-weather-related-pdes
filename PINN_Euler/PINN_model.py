import numpy as np
from numpy.core.fromnumeric import take
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.optimizers import Adam

from typing import List, Tuple


class PINNModel(tf.keras.Model):
    # initialize the model
    def __init__(
        self,
        layers: List[int],
        bound: Tuple[float, float],
    ) -> None:
        super(PINNModel, self).__init__()
        # loading parameters
        self.lower_bound, self.upper_bound = bound
        # set up network
        self.layers_list = layers
        self.net = self.create_network(layers)

    def create_network(self, layers: List[int]):
        hidden_layers = [
            Dense(
                layer,
                activation=Lambda(lambda x: tf.sin(x)),
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
        X = self.normalize(X)
        for layer in self.net:
            X = layer(X)
        return X

    @tf.function
    def normalize(self, x):
        return 2 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1


class PINN_NS(PINNModel):
    def __init__(
        self, layers: List[int], bound: Tuple[float, float], Reynold: float
    ) -> None:
        super().__init__(Reynold, layers, bound)
        self.Reynold = Reynold
        self.nu_v = 1 / self.Reynold

    # compute eqn
    @tf.function
    def compute_NS_eqn(self, t, x, y):
        with tf.GradientTape(persistent=True) as t2:
            t2.watch([t, x, y])
            with tf.GradientTape(persistent=True) as t1:
                t1.watch([t, x, y])
                uvp = self.call(t, x, y)
                u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
            # first order gradients
            u_t = t1.gradient(u, t)
            u_x = t1.gradient(u, x)
            u_y = t1.gradient(u, y)
            v_t = t1.gradient(v, t)
            v_x = t1.gradient(v, x)
            v_y = t1.gradient(v, y)
            p_x = t1.gradient(p, x)
            p_y = t1.gradient(p, y)
            del t1
        # second order gradients
        u_xx = t2.gradient(u_x, x)
        u_yy = t2.gradient(u_y, y)
        v_xx = t2.gradient(v_x, x)
        v_yy = t2.gradient(v_y, y)
        del t2
        eq1 = u_t + (u * u_x + v * u_y) + p_x - (self.nu_v) * (u_xx + u_yy)
        eq2 = v_t + (u * v_x + v * v_y) + p_y - (self.nu_v) * (v_xx + v_yy)
        eq3 = u_x + v_y
        return eq1, eq2, eq3

    # evaluation
    @tf.function
    def predict_uvp(self, t, x, y) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        uvp = self.call(t, x, y)
        u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]
        return u, v, p

    @tf.function
    def predict_vorticity(self, t, x, y) -> tf.Tensor:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([t, x, y])
            uvp = self.call(t, x, y)
            u, v = uvp[:, 0], uvp[:, 1]
        u_y = tape.gradient(u, y)
        v_x = tape.gradient(v, x)
        del tape
        OmegaZ = v_x - u_y
        return OmegaZ[:, 0]


class PINN_Euler(PINNModel):
    def __init__(
        self, layers: List[int], bound: Tuple[float, float], gamma: float
    ) -> None:
        super().__init__(layers, bound)
        self.gamma = gamma

    @tf.function
    def compute_Euler_eqn(self, t, x, y):
        """
        output: [rho, rho*u, rho*v, E]
        E = 0.5*rho*(u^2+v^2)+rho*e
        p = (gamma - 1)*rho*e
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([t, x, y])
            ruve = self.call(t, x, y)
            rho, rho_u, rho_v, E = ruve[:, 0:1], ruve[:, 1:2], ruve[:, 2:3], ruve[:, 3:4]
            # calculate intermediate variables
            u, v = rho_u / rho, rho_v / rho
            rho_u2, rho_v2 = rho_u * u, rho_v * v
            p = (self.gamma - 1) * (E - 0.5 * (rho_u2 + rho_v2))
            u1, f1, g1 = rho, rho_u, rho_v
            u2, f2, g2 = rho_u, rho_u2 + p, rho * u * v
            u3, f3, g3 = rho_v, rho * u * v, rho_v2 + p
            u4, f4, g4 = E, (E + p) * u, (E + p) * v
            # calculate gradients
            u1_t = tape.gradient(u1, t)
            f1_x = tape.gradient(f1, x)
            g1_y = tape.gradient(g1, y)
            u2_t = tape.gradient(u2, t)
            f2_x = tape.gradient(f2, x)
            g2_y = tape.gradient(g2, y)
            u3_t = tape.gradient(u3, t)
            f3_x = tape.gradient(f3, x)
            g3_y = tape.gradient(g3, y)
            u4_t = tape.gradient(u4, t)
            f4_x = tape.gradient(f4, x)
            g4_y = tape.gradient(g4, y)
        del tape
        eq1 = u1_t + f1_x + g1_y
        eq2 = u2_t + f2_x + g2_y
        eq3 = u3_t + f3_x + g3_y
        eq4 = u4_t + f4_x + g4_y
        return eq1, eq2, eq3, eq4

    # evaluation
    @tf.function
    def predict_ruve(self, t, x, y) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        ruve = self.call(t, x, y)
        rho, rho_u, rho_v, E = ruve[:, 0:1], ruve[:, 1:2], ruve[:, 2:3], ruve[:, 3:4]
        # calculate intermediate variables
        u, v = rho_u / rho, rho_v / rho
        rho_u2, rho_v2 = rho_u * u, rho_v * v
        p = (self.gamma - 1) * (E - 0.5 * (rho_u2 + rho_v2))

        return rho, u, v, p
