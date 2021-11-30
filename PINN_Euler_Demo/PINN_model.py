import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.optimizers import Adam

from typing import List, Tuple


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
        for layer in self.net:
            X = layer(X)
        return X


class PINN_NS(PINNModel):
    def __init__(self, layers: List[int], Reynold: float) -> None:
        super().__init__(layers)
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
        self, layers: List[int], gamma: float, weight: Tuple[float, float]
    ) -> None:
        super().__init__(layers)
        self.gamma = gamma
        self.lambda_eqn, self.lambda_ic, self.lambda_bc = weight
        self.optimizer = Adam()
        self.epsilon = 1e-6

    @tf.function
    def compute_Euler_eqn(self, t, x, y):
        """
        output: [rho, u, v, p]
        E = 0.5*rho*(u^2+v^2)+rho*e
        p = (gamma - 1)*rho*e
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([t, x, y])
            ruvp = self.call(t, x, y)
            rho, u, v, p = (
                ruvp[:, 0:1],
                ruvp[:, 1:2],
                ruvp[:, 2:3],
                ruvp[:, 3:4],
            )
            # calculate intermediate variables
            rho_e = p / (self.gamma - 1)
            E = 0.5 * rho * (u ** 2 + v ** 2) + rho_e
            u1, f1, g1 = rho, rho * u, rho * v
            u2, f2, g2 = rho * u, rho * u ** 2 + p, rho * u * v
            u3, f3, g3 = rho * v, rho * u * v, rho * v ** 2 + p
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

    @tf.function
    def predict_ruvp(self, t, x, y):
        ruvp = self.call(t, x, y)
        rho, u, v, p = ruvp[:, 0:1], ruvp[:, 1:2], ruvp[:, 2:3], ruvp[:, 3:4]
        return rho, u, v, p

    @tf.function
    def train_step(self, X_res, X_ic, X_bc):
        """
        TODO
        train for a single batch
        X_res: [t_res, x_res, y_res]
        X_ic: [t_res, x_res, y_res, nx_res, ny_res, u_res, v_res]
        X_bc: [X_lr, X_du]
        X_lr: [t_lr, x_l, y_l, t_lr, x_r, y_r]
        X_du: [t_du, x_d, y_d, t_du, x_u, y_u]
        return: loss, loss_supervised, loss_eqn_momentum_x, loss_eqn_momentum_y, loss_eqn_mass
        """
        # load data
        t_res, x_res, y_res = X_res[:, 0:1], X_res[:, 1:2], X_res[:, 2:3]
        t_ic, x_ic, y_ic = X_ic[:, 0:1], X_ic[:, 1:2], X_ic[:, 2:3]
        rho, u, v, p = X_ic[:, 3:4], X_ic[:, 4:5], X_ic[:, 5:6], X_ic[:, 6:7]
        # load bc
        X_lr, X_du = X_bc
        t_lr, x_l, y_l, t_lr, x_r, y_r = (
            X_lr[:, 0:1],
            X_lr[:, 1:2],
            X_lr[:, 2:3],
            X_lr[:, 3:4],
            X_lr[:, 4:5],
            X_lr[:, 5:6],
        )
        t_du, x_d, y_d, t_du, x_u, y_u = (
            X_du[:, 0:1],
            X_du[:, 1:2],
            X_du[:, 2:3],
            X_du[:, 3:4],
            X_du[:, 4:5],
            X_du[:, 5:6],
        )

        # take gradient and BP
        with tf.GradientTape() as tape:
            # Euler loss
            (
                eqn_mass,
                eqn_momentum_x,
                eqn_momentum_y,
                eqn_energy,
            ) = self.compute_Euler_eqn(t_res, x_res, y_res)
            loss_eqn_mass = tf.math.reduce_mean(eqn_mass ** 2)
            loss_eqn_momentum_x = tf.math.reduce_mean(eqn_momentum_x ** 2)
            loss_eqn_momentum_y = tf.math.reduce_mean(eqn_momentum_y ** 2)
            loss_eqn_energy = tf.math.reduce_mean(eqn_energy ** 2)
            loss_Euler = (
                loss_eqn_mass
                + loss_eqn_momentum_x
                + loss_eqn_momentum_y
                + loss_eqn_energy
            )
            # initial condition loss
            rho_pred, u_pred, v_pred, p_pred = self.predict_ruvp(t_ic, x_ic, y_ic)
            loss_ic = tf.math.reduce_mean(
                (rho_pred - rho) ** 2
                + (u_pred - u) ** 2
                + (v_pred - v) ** 2
                + (p_pred - p) ** 2
            )
            # periodic bpundary condition
            rho_l, u_l, v_l, p_l = self.predict_ruvp(t_lr, x_l, y_l)
            rho_r, u_r, v_r, p_r = self.predict_ruvp(t_lr, x_r, y_r)
            rho_d, u_d, v_d, p_d = self.predict_ruvp(t_du, x_d, y_d)
            rho_u, u_u, v_u, p_u = self.predict_ruvp(t_du, x_u, y_u)
            loss_bc = tf.math.reduce_mean(
                (rho_l - rho_r) ** 2
                + (u_l - u_r) ** 2
                + (v_l - v_r) ** 2
                + (p_l - p_r) ** 2
            ) + tf.math.reduce_mean(
                (rho_d - rho_u) ** 2
                + (u_d - u_u) ** 2
                + (v_d - v_u) ** 2
                + (p_d - p_u) ** 2
            )
            # total loss
            loss = (
                self.lambda_eqn * loss_Euler
                + self.lambda_ic * loss_ic
                + self.lambda_bc * loss_bc
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return (
            loss,
            loss_eqn_mass,
            loss_eqn_momentum_x,
            loss_eqn_momentum_y,
            loss_eqn_energy,
            loss_ic,
            loss_bc,
        )
