import imp
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import scipy.io as sio

from typing import List, Tuple

from model.PINN_model import PINNModel
from utils import InverseTimeDecay, boundary_condition


class PINN_Euler(PINNModel):
    def __init__(
        self,
        layers: List[int],
        gamma: float,
        scale: Tuple[float, float, float, float],
        boundarys: List[float],
        boundary_thickness: float,
        weight: Tuple[float, float],
        decay_step: int,
    ) -> None:
        super().__init__(layers)
        self.gamma = gamma
        self.rho_scale, self.u_scale, self.v_scale, self.p_scale = scale
        self.boundarys = boundarys
        self.boundary_thickness = boundary_thickness
        self.lambda_eqn, self.lambda_ic, self.lambda_bc = weight
        self.optimizer = Adam(learning_rate=InverseTimeDecay(0.001, decay_step, 0.5))
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
            rho, u, v, p = self.predict_ruvp(t, x, y)
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
        X_bc: [Xc_lr, Xc_du]
        Xc_lr: [tc_lr, xc_l, yc_l, tc_lr, xc_r, yc_r]
        Xc_du: [tc_du, xc_d, yc_d, tc_du, xc_u, yc_u]
        return: loss, loss_supervised, loss_eqn_momentum_x, loss_eqn_momentum_y, loss_eqn_mass
        """
        # load data
        t_res, x_res, y_res = X_res[:, 0:1], X_res[:, 1:2], X_res[:, 2:3]
        t_ic, x_ic, y_ic = X_ic[:, 0:1], X_ic[:, 1:2], X_ic[:, 2:3]
        rho, u, v, p = X_ic[:, 3:4], X_ic[:, 4:5], X_ic[:, 5:6], X_ic[:, 6:7]

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

            loss_bc_inflow, loss_bc_reflective = self.reflective_boundary_condition(
                X_bc
            )
            loss_bc = loss_bc_inflow + loss_bc_reflective
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
            loss_bc_inflow,
            loss_bc_reflective,
        )

    @tf.function
    def reflective_boundary_condition(self, X_bc):
        # load bc
        X_lr, X_du, Xc_lr, Xc_du = X_bc

        def unpack_X_bc(X):
            assert X.shape[1] == 6, "Not correct X_bc shape"
            return X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4], X[:, 4:5], X[:, 5:6]

        t_lr, x_l, y_l, t_lr, x_r, y_r = unpack_X_bc(X_lr)
        t_du, x_d, y_d, t_du, x_u, y_u = unpack_X_bc(X_du)
        tc_lr, xc_l, yc_l, tc_lr, xc_r, yc_r = unpack_X_bc(Xc_lr)
        tc_du, xc_d, yc_d, tc_du, xc_u, yc_u = unpack_X_bc(Xc_du)

        bt = self.boundary_thickness

        # inner boundary condition
        # inner left boundary
        rhoc_lp, uc_lp, vc_lp, pc_lp = self.predict_ruvp(tc_lr, xc_l + bt, yc_l)
        rhoc_lm, uc_lm, vc_lm, pc_lm = self.predict_ruvp(tc_lr, xc_l - bt, yc_l)
        # loss_ref_l = tf.math.reduce_mean((uc_lm + tf.stop_gradient(uc_lp)) ** 2)
        loss_ref_l = tf.math.reduce_mean(uc_lm ** 2)
        # inner right boundary
        rhoc_rp, uc_rp, vc_rp, pc_rp = self.predict_ruvp(tc_lr, xc_r - bt, yc_r)
        rhoc_rm, uc_rm, vc_rm, pc_rm = self.predict_ruvp(tc_lr, xc_r + bt, yc_r)
        # loss_ref_r = tf.math.reduce_mean((uc_rm + tf.stop_gradient(uc_rp)) ** 2)
        loss_ref_r = tf.math.reduce_mean(uc_rm ** 2)
        # inner down boundary
        rhoc_dp, uc_dp, vc_dp, pc_dp = self.predict_ruvp(tc_du, xc_d, yc_d + bt)
        rhoc_dm, uc_dm, vc_dm, pc_dm = self.predict_ruvp(tc_du, xc_d, yc_d - bt)
        # loss_ref_d = tf.math.reduce_mean((vc_dm + tf.stop_gradient(vc_dp)) ** 2)
        loss_ref_d = tf.math.reduce_mean(vc_dm ** 2)
        # inner up boundary
        rhoc_up, uc_up, vc_up, pc_up = self.predict_ruvp(tc_du, xc_u, yc_u - bt)
        rhoc_um, uc_um, vc_um, pc_um = self.predict_ruvp(tc_du, xc_u, yc_u + bt)
        # loss_ref_u = tf.math.reduce_mean((vc_um + tf.stop_gradient(vc_up)) ** 2)
        loss_ref_u = tf.math.reduce_mean(vc_um ** 2)
        loss_bc_inner = loss_ref_l + loss_ref_r + loss_ref_d + loss_ref_u

        # constant flow from left
        rho_boundary, u_boundary, v_boundary, p_boundary = self.boundarys
        rho_l, u_l, v_l, p_l = self.predict_ruvp(t_lr, x_l, y_l)
        rho_d, u_d, v_d, p_d = self.predict_ruvp(t_du, x_d, y_d)
        rho_u, u_u, v_u, p_u = self.predict_ruvp(t_du, x_u, y_u)
        # loss_bc_inflow = 20 * (
        #     tf.math.reduce_mean((u_l - u_boundary) ** 2)
        #     + tf.math.reduce_mean((v_l - v_boundary) ** 2)
        #     + tf.math.reduce_mean((rho_l - rho_boundary) ** 2)
        #     + tf.math.reduce_mean((p_l - p_boundary) ** 2)
        # ) + 5 * (
        #     tf.math.reduce_mean((u_d - u_boundary) ** 2)
        #     + tf.math.reduce_mean((u_u - u_boundary) ** 2)
        # )
        # return loss_bc_inflow, 10 * loss_bc_inner
        loss_bc_inflow_left = (
            tf.math.reduce_mean((u_l - u_boundary) ** 2)
            + tf.math.reduce_mean((v_l - v_boundary) ** 2)
            + tf.math.reduce_mean((rho_l - rho_boundary) ** 2)
            + tf.math.reduce_mean((p_l - p_boundary) ** 2)
        )
        loss_bc_inflow_up = (
            tf.math.reduce_mean((u_u - u_boundary) ** 2)
            + tf.math.reduce_mean((v_u - v_boundary) ** 2)
            + tf.math.reduce_mean((rho_u - rho_boundary) ** 2)
            + tf.math.reduce_mean((p_u - p_boundary) ** 2)
        )
        loss_bc_inflow_down = (
            tf.math.reduce_mean((u_d - u_boundary) ** 2)
            + tf.math.reduce_mean((v_d - v_boundary) ** 2)
            + tf.math.reduce_mean((rho_d - rho_boundary) ** 2)
            + tf.math.reduce_mean((p_d - p_boundary) ** 2)
        )
        loss_bc_inflow = loss_bc_inflow_left + loss_bc_inflow_up + loss_bc_inflow_down
        return loss_bc_inflow, loss_bc_inner


def evaluate_and_save(
    model: PINN_Euler,
    X_final: np.ndarray,
    mask: np.ndarray,
    shape: Tuple[int, int],
    loss_history: list,
    save_path: str,
):
    H, W = shape
    t_f, x_f, y_f = X_final[:, 0:1], X_final[:, 1:2], X_final[:, 2:3]
    rho_pred, u_pred, v_pred, p_pred = model.predict_ruvp(t_f, x_f, y_f)
    rho = rho_pred.numpy().reshape(H, W) * mask
    u = u_pred.numpy().reshape(H, W) * mask
    v = v_pred.numpy().reshape(H, W) * mask
    p = p_pred.numpy().reshape(H, W) * mask
    sio.savemat(
        save_path,
        {"rho": rho, "u": u, "v": v, "p": p, "loss": loss_history},
    )
