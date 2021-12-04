import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import scipy.io as sio

from typing import List, Tuple

from model.PINN_model import PINNModel
from utils import InverseTimeDecay


class PINN_NS(PINNModel):
    def __init__(
        self,
        layers: List[int],
        Reynold: float,
        u_boundary: float,
        weight: Tuple[float, float, float],
        decay_step: int,
    ) -> None:
        super().__init__(layers)
        self.Reynold = Reynold
        self.nu_v = 1 / self.Reynold
        self.u_boundary = u_boundary
        self.lambda_eqn, self.lambda_ic, self.lambda_bc = weight
        self.optimizer = Adam(learning_rate=InverseTimeDecay(0.001, decay_step, 0.5))

    @tf.function
    def predict_uvp(self, t, x, y):
        uvp = self.call(t, x, y)
        u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
        return u, v, p

    # compute eqn
    @tf.function
    def compute_NS_eqn(self, t, x, y):
        with tf.GradientTape(persistent=True) as t2:
            t2.watch([t, x, y])
            with tf.GradientTape(persistent=True) as t1:
                t1.watch([t, x, y])
                u, v, p = self.predict_uvp(t, x, y)
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
        eq1 = u_x + v_y
        eq2 = u_t + (u * u_x + v * u_y) + p_x - (self.nu_v) * (u_xx + u_yy)
        eq3 = v_t + (u * v_x + v * v_y) + p_y - (self.nu_v) * (v_xx + v_yy)
        return eq1, eq2, eq3

    # compute eqn
    @tf.function
    def compute_Inviscid_NS_eqn(self, t, x, y):
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([t, x, y])
            u, v, p = self.predict_uvp(t, x, y)
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
        eq1 = u_x + v_y
        eq2 = u_t + (u * u_x + v * u_y) + p_x
        eq3 = v_t + (u * v_x + v * v_y) + p_y
        return eq1, eq2, eq3

    @tf.function
    def train_step(self, X_res, X_ic, X_bc):
        """
        TODO
        train for a single batch
        X_res: [t_res, x_res, y_res]
        X_ic: [t_res, x_res, y_res, u, v, p]
        X_bc: [Xc_lr, Xc_du]
        Xc_lr: [tc_lr, xc_l, yc_l, tc_lr, xc_r, yc_r]
        Xc_du: [tc_du, xc_d, yc_d, tc_du, xc_u, yc_u]
        return: loss, loss_supervised, loss_eqn_momentum_x, loss_eqn_momentum_y, loss_eqn_mass
        """
        # load data
        t_res, x_res, y_res = X_res[:, 0:1], X_res[:, 1:2], X_res[:, 2:3]
        t_ic, x_ic, y_ic = X_ic[:, 0:1], X_ic[:, 1:2], X_ic[:, 2:3]
        u, v, p = X_ic[:, 3:4], X_ic[:, 4:5], X_ic[:, 5:6]

        # take gradient and BP
        with tf.GradientTape() as tape:
            # NS loss
            (
                eqn_mass,
                eqn_momentum_x,
                eqn_momentum_y,
            ) = self.compute_Inviscid_NS_eqn(t_res, x_res, y_res)
            loss_eqn_mass = tf.math.reduce_mean(eqn_mass ** 2)
            loss_eqn_momentum_x = tf.math.reduce_mean(eqn_momentum_x ** 2)
            loss_eqn_momentum_y = tf.math.reduce_mean(eqn_momentum_y ** 2)
            loss_NS = loss_eqn_mass + loss_eqn_momentum_x + loss_eqn_momentum_y
            # initial condition loss
            u_pred, v_pred, p_pred = self.predict_uvp(t_ic, x_ic, y_ic)
            loss_ic = tf.math.reduce_mean(
                (u_pred - u) ** 2 + (v_pred - v) ** 2 + (p_pred - p) ** 2
            )
            # BC loss
            loss_bc_inflow, loss_bc_inner = self.boundary_condition(X_bc)
            loss_bc = loss_bc_inflow + 2 * loss_bc_inner
            # total loss
            loss = (
                self.lambda_eqn * loss_NS
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
            loss_ic,
            loss_bc_inflow,
            loss_bc_inner,
        )

    @tf.function
    def boundary_condition(self, X_bc):
        # load bc
        X_lr, X_du, Xc_lr, Xc_du = X_bc

        def unpack_X_bc(X):
            assert X.shape[1] == 6, "Not correct X_bc shape"
            return X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4], X[:, 4:5], X[:, 5:6]

        t_lr, x_l, y_l, t_lr, x_r, y_r = unpack_X_bc(X_lr)
        t_du, x_d, y_d, t_du, x_u, y_u = unpack_X_bc(X_du)
        tc_lr, xc_l, yc_l, tc_lr, xc_r, yc_r = unpack_X_bc(Xc_lr)
        tc_du, xc_d, yc_d, tc_du, xc_u, yc_u = unpack_X_bc(Xc_du)

        # no penetrate boundary condition
        # inner left boundary
        uc_l, vc_l, pc_l = self.predict_uvp(tc_lr, xc_l, yc_l)
        loss_inner_l = tf.math.reduce_mean(uc_l ** 2)
        # inner right boundary
        uc_r, vc_r, pc_r = self.predict_uvp(tc_lr, xc_r, yc_r)
        loss_inner_r = tf.math.reduce_mean(uc_r ** 2)
        # inner down boundary
        uc_d, vc_d, pc_d = self.predict_uvp(tc_du, xc_d, yc_d)
        loss_inner_d = tf.math.reduce_mean(vc_d ** 2)
        # inner up boundary
        uc_u, vc_u, pc_u = self.predict_uvp(tc_du, xc_u, yc_u)
        loss_inner_u = tf.math.reduce_mean(vc_u ** 2)
        loss_bc_inner = loss_inner_l + loss_inner_r + loss_inner_d + loss_inner_u

        # constant flow from left
        u_l, v_l, p_l = self.predict_uvp(t_lr, x_l, y_l)
        u_d, v_d, p_d = self.predict_uvp(t_du, x_d, y_d)
        u_u, v_u, p_u = self.predict_uvp(t_du, x_u, y_u)
        loss_bc_inflow = (
            tf.math.reduce_mean((u_l - self.u_boundary) ** 2)
            + tf.math.reduce_mean((u_d - self.u_boundary) ** 2)
            + tf.math.reduce_mean((u_u - self.u_boundary) ** 2)
        )
        return loss_bc_inflow, loss_bc_inner

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


def evaluate_and_save(
    model: PINN_NS,
    X_final: np.ndarray,
    mask: np.ndarray,
    shape: Tuple[int, int],
    loss_history: list,
    save_path: str,
):
    H, W = shape
    t_f, x_f, y_f = X_final[:, 0:1], X_final[:, 1:2], X_final[:, 2:3]
    u_pred, v_pred, p_pred = model.predict_uvp(t_f, x_f, y_f)
    u = u_pred.numpy().reshape(H, W) * mask
    v = v_pred.numpy().reshape(H, W) * mask
    p = p_pred.numpy().reshape(H, W) * mask
    sio.savemat(save_path, {"u": u, "v": v, "p": p, "loss": loss_history})
