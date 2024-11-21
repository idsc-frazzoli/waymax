"""
Implementation of the bicycle (acceleration, steering) dynamics model.

This action space always uses the [-1.0, 1.0] as the range for acceleration
and steering commands to be consistent with other RL training pipeline since
many algorithms' hyperparameters are tuned based on this assumption.
The actual acceleration and steering command range can still be specified by `max_accel`
and `max_steering` in the class definition function.
"""

from abc import ABC
from inspect import isclass
import jax
import jax.numpy as jnp
import numpy as np
from dm_env import specs

from waymax import datatypes
from waymax.dynamics import abstract_dynamics
from waymax.utils import geometry
from waymax.utils.gokart_config import (
    GoKartGeometry,
    TricycleParams,
    TricycleDynamicsType,
    PajieckaParams,
    PacejkaParamsIgnition,
)

DynamicsModel = abstract_dynamics.DynamicsModel
G = 9.81  # gravitational acceleration m/s^2
RHO = 1.249512  # air density kg/m^3


# Call the factory function to create the correct tricycle dynamics model
# Implemented in this way to avoid breaking existing code
def TricycleModel(
    gk_geometry: GoKartGeometry,
    model_params: TricycleParams,
    paj_params: PajieckaParams,
    dt: float = 0.1,
    normalize_actions: bool = True,
):
    if model_params.dynamics_model == TricycleDynamicsType.FORCES:
        model = ForcesTricycleModel(gk_geometry, model_params, paj_params, dt, normalize_actions)
    elif model_params.dynamics_model == TricycleDynamicsType.IGNITION:
        model = IgnitionTricycleModel(gk_geometry, model_params, paj_params, dt, normalize_actions)
    elif model_params.dynamics_model == TricycleDynamicsType.ORIGINAL:
        model = OriginalTricycleModel(gk_geometry, model_params, paj_params, dt, normalize_actions)
    else:
        raise ValueError(f"Unknown dynamics model: {model_params.dynamics_model}")

    print(f"Using dynamics model: {model_params.dynamics_model.name} ({model.__class__.__name__})")
    return model


class TricycleModelABC(DynamicsModel):
    """Dynamics model using acceleration and steering curvature for control."""

    def __init__(
        self,
        gk_geometry: GoKartGeometry,
        model_params: TricycleParams,
        paj_params: PajieckaParams,
        dt: float = 0.1,
        normalize_actions: bool = True,
    ):
        """Initializes the bounds of the action space.

        Args:
          dt: The time length per step used in the simulator in seconds.
          max_accel: The maximum acceleration magnitude.
          max_steering: The maximum steering curvature magnitude, which is the
            inverse of the turning radius (the minimum radius of available space
            required for that vehicle to make a circular turn).
          normalize_actions: Whether to normalize the action range to [-1,1] or not.
            By default, it uses the unnormalized range in order to train with RL,
            such as with ACME. Ideally we should normalize the ranges.
        """

        # super().__init__()
        self._gk_geometry = gk_geometry
        self._model_params = model_params
        self._paj_params = paj_params
        self._dt = dt
        self._normalize_actions = normalize_actions

    def action_spec(self) -> specs.BoundedArray:
        """
        Action spec for the acceleration steering continuous action space.
        """
        action_shape = (3,)
        if self._normalize_actions:
            return specs.BoundedArray(
                # last dim: (acceleration, steering)
                shape=action_shape,
                dtype=np.float32,
                minimum=np.ones(action_shape) * -1.0,
                maximum=np.ones(action_shape),
            )
        else:
            return specs.BoundedArray(
                # last dim: (steering)
                shape=action_shape,
                dtype=np.float32,
                minimum=np.array(
                    [-self._model_params.max_steering, -self._model_params.max_accel, -self._model_params.max_accel]
                ),
                maximum=np.array(
                    [self._model_params.max_steering, self._model_params.max_accel, self._model_params.max_accel]
                ),
            )

    def _clip_values(self, action_array: jax.Array) -> jax.Array:
        """Clip action values to be within the allowable ranges."""
        return jnp.clip(action_array, jnp.asarray(self.action_spec().minimum), jnp.asarray(self.action_spec().maximum))

    @jax.named_scope("TricycleModel.compute_update")
    def compute_update(
        self,
        action: datatypes.Action,
        trajectory: datatypes.GokartTrajectory,
    ) -> datatypes.GoKartTrajectoryUpdate:
        """Computes the pose and velocity updates at timestep.

        Args:
          action: Actions of shape (..., num_objects) containing acceleration and
            steering controls.
          trajectory: Trajectory to be updated. Has shape of (..., num_objects,
            num_timesteps=1).

        Returns:
          The trajectory update for timestep of shape
            (..., num_objects, num_timesteps=1).
        """
        # x = trajectory.x
        # y = trajectory.y
        # vel_x = trajectory.vel_x
        # vel_y = trajectory.vel_y
        # yaw = trajectory.yaw
        # speed = jnp.sqrt(trajectory.vel_x**2 + trajectory.vel_y**2)

        # Shape: (..., num_objects, 2)
        # action_array = self._clip_values(action.data)
        # accel, steering = jnp.split(action_array, 2, axis=-1)
        # if self._normalize_actions:
        #   accel = accel * self._max_accel
        #   steering = steering * self._max_steering
        t = self._dt

        x = trajectory.x  # shape (..., num_objects, num_timesteps=1)
        y = trajectory.y
        vel_x = trajectory.vel_x
        vel_y = trajectory.vel_y
        yaw = trajectory.yaw
        yaw_rate = trajectory.yaw_rate
        acc_x = trajectory.acc_x
        acc_y = trajectory.acc_y
        # yaw_rate = jnp.zeros_like(vel_x)
        state = jnp.concatenate((x, y, vel_x, vel_y, yaw, yaw_rate, acc_x, acc_y), axis=-1)

        action_clipped = self._clip_values(action.data)

        # Vectorize _RK4_update function along batch and num_objects dimensions
        if len(x.shape) == 2:  # x shape (num_objects, num_timesteps=1)
            rk4_vmap = jax.vmap(self._RK4_update, in_axes=(0, 0, None))
        elif len(x.shape) == 3:  # x shape (batch_size, num_objects, num_timesteps=1)
            rk4_vmap = jax.vmap(jax.vmap(self._RK4_update, in_axes=(0, 0, None)), in_axes=(0, 0, None))
        else:
            raise ValueError("Invalid shape for x: {}".format(x.shape))

        new_states = rk4_vmap(action_clipped, state, t)

        return datatypes.GoKartTrajectoryUpdate(
            x=new_states[..., 0:1],
            y=new_states[..., 1:2],
            yaw=geometry.wrap_yaws(new_states[..., 4:5]),
            vel_x=new_states[..., 2:3],
            vel_y=new_states[..., 3:4],
            yaw_rate=new_states[..., 5:6],
            acc_x=new_states[..., 6:7],
            acc_y=new_states[..., 7:8],
            valid=trajectory.valid & action.valid,
        )

    def _dynamics(self, action: jax.Array, state: jnp.ndarray) -> jnp.ndarray:
        if self._normalize_actions:
            # scale back up if actions are normalized
            raw_act_min = -jnp.array(
                [self._model_params.max_steering, self._model_params.max_accel, self._model_params.max_accel]
            )
            raw_act_max = jnp.array(
                [self._model_params.max_steering, self._model_params.max_accel, self._model_params.max_accel]
            )
            act_min = self.action_spec().minimum
            act_max = self.action_spec().maximum
            # convert normalized action [-1,1] to a real world action (eg acceleration [-6,-6])
            action = raw_act_min + (raw_act_max - raw_act_min) * (action - act_min) / (act_max - act_min)

        return self._dynamics_model(action, state)

    def _dynamics_model(self, action: jax.Array, state: jnp.ndarray):
        raise NotImplementedError("Subclasses must implement this method.")

    def _ackermann_mapping(self, steering: float, two_wheels: bool = False) -> float:
        """Maps angle of steerig wheel to the steering angle of front wheel."""
        if two_wheels:
            return (self._ackermann_mapping_left(steering) + self._ackermann_mapping_right(steering)) / 2
        else:
            return -0.065 * steering * steering * steering + 0.45 * steering
            # return -0.04253 * steering * steering * steering + 4.455e-05 * steering * steering + 0.4039 * steering

    def _ackermann_mapping_left(self, steering: float) -> float:
        """Maps angle of steerig wheel to the steering angle of front left wheel."""
        return -0.0355 * steering * steering * steering - 0.0455 * steering * steering + 0.36 * steering

    def _ackermann_mapping_right(self, steering: float) -> float:
        """Maps angle of steerig wheel to the steering angle of front right wheel."""
        return -0.0355 * steering * steering * steering + 0.0455 * steering * steering + 0.36 * steering

    def _RK4_update(self, action, state, dt):
        """
        Runge-Kutta 4th order integration
        """
        # acc_x and acc_y (index 6 and 7) are already defined as the next state
        # values and are not "derivatives" of the state, so for them we
        # reset the Runga-Kutta output values
        k1 = self._dynamics(action, state)
        acc_x = k1[6]
        acc_y = k1[7]
        k1 = k1.at[6].set(state[6])
        k1 = k1.at[7].set(state[7])
        k2 = self._dynamics(action, state + dt / 2 * k1)
        k2 = k2.at[6].set(state[6])
        k2 = k2.at[7].set(state[7])
        k3 = self._dynamics(action, state + dt / 2 * k2)
        k3 = k3.at[6].set(state[6])
        k3 = k3.at[7].set(state[7])
        k4 = self._dynamics(action, state + dt * k3)
        k3 = k3.at[6].set(state[6])
        k3 = k3.at[7].set(state[7])

        next_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        next_state = next_state.at[6].set(acc_x)
        next_state = next_state.at[7].set(acc_y)
        
        

        return next_state

    def _euler_forward(self, action, state, dt):
        """
        Euler forward integration
        """
        state_derivative = self._dynamics(action, state)
        next_state = state + dt * state_derivative
        # acc_x and acc_y (index 6 and 7) are already defined as the next state
        # values and are not "derivatives" of the state, so for them we
        # reset the euler output values
        next_state = next_state.at[6].set(state_derivative[6])
        next_state = next_state.at[7].set(state_derivative[7])

        return next_state

    def inverse(self):
        raise NotImplementedError


class ForcesTricycleModel(TricycleModelABC):

    def __init__(
        self,
        gk_geometry: GoKartGeometry,
        model_params: TricycleParams,
        paj_params: PajieckaParams,
        dt: float = 0.1,
        normalize_actions: bool = True,
    ):
        super().__init__(gk_geometry, model_params, paj_params, dt, normalize_actions)

    def _dynamics_model(
        self,
        action: jax.Array,
        state: jnp.ndarray,
    ):
        """
        Note: all dynamics are normalized w.r.t. the normal force
        hence the name *_acc instead of *_force
        Action: beta: steering wheel angle
                AB_L: left rear wheel acceleration [-1, 1]
                AB_R: right rear wheel acceleration [-1, 1]
                braking: braking force (not used)?
        State:  x: x position
                y: y position
                vel_x: x velocity
                vel_y: y velocity
                acc_x: x acceleration
                acc_y: y acceleration
                yaw: yaw angle
                yaw_rate: yaw rate
                yaw_acc: yaw acceleration
        """

        # beta, AB_L, AB_R = jnp.split(action_array, 3, axis=-1)
        beta, AB_L, AB_R = action
        AB = AB_L + AB_R
        tv = AB_R - AB_L

        x, y, v_x, v_y, yaw, vrot_z, a_x, a_y = state  # float shape ()

        reg = jnp.array([self._model_params.REG_])  # self._get_reg_from_velocity(v_x)

        # region front wheel acc
        # front wheel angle
        delta = self._ackermann_mapping(beta, two_wheels=False)  # steering angle

        # velocities at front wheel (Marc Heim, (2.43))
        vel1 = jnp.array([v_x, v_y + self._gk_geometry.l1 * vrot_z])  # go kart frame
        # Tire Forces (velocity in wheels reference frame)
        rot_delta = geometry.rotation_matrix(delta)

        # # Front wheel velocity in wheel reference frame (Adaption from Marc Heim (2.82f))
        v_frontaxle = rot_delta.T @ vel1  # front tyre frame
        # friction coefficient front
        mu_front = self._mu_y_front(v_frontaxle[1], v_frontaxle[0], reg=reg)

        # longitudinal load transfer
        load_transfer = self._gk_geometry.h * AB
        fz_f = -(G * self._gk_geometry.l2 - load_transfer) * self._gk_geometry.m / self._gk_geometry.l
        fz_r = -(G * self._gk_geometry.l1 + load_transfer) * self._gk_geometry.m / self._gk_geometry.l

        # Longitudinal and lateral force in kart frame at the front wheel
        # jax.debug.print("shapes: mu_front = {}, fz_f = {}", mu_front.shape, fz_f.shape)
        # jax.debug.print("mu_front = {}, fz_f = {}\n", mu_front, fz_f)
        f_front_wheel = rot_delta @ jnp.array([0.0, mu_front * fz_f])
        f_x_front = f_front_wheel[0]
        f_y_front = f_front_wheel[1]

        # Longitudinal force from both rear wheels (f_x_left+f_x_right)
        f_x_rear = AB * self._gk_geometry.m

        # Back Wheel longitudinal and lateral velocities
        v_x_backaxle_l = v_x - self._gk_geometry.w2 / 2 * vrot_z
        v_x_backaxle_r = v_x + self._gk_geometry.w2 / 2 * vrot_z
        v_y_backaxle = v_y - self._gk_geometry.l2 * vrot_z

        # Lateral force from left rear wheel
        f_y_left_rear = self._mu_y_rear(v_y_backaxle, v_x_backaxle_l, AB_L, reg=reg) * fz_r / 2
        # Lateral force from right rear wheel
        f_y_right_rear = self._mu_y_rear(v_y_backaxle, v_x_backaxle_r, AB_R, reg=reg) * fz_r / 2

        # Torque vectoring
        tv_torque = tv * self._gk_geometry.m * self._gk_geometry.w2 / 2

        # ------ Gokart accelerations
        # Rotational Acceleration of the kart
        rotacc_z = (
            tv_torque + f_y_front * self._gk_geometry.l1 - (f_y_right_rear + f_y_left_rear) * self._gk_geometry.l2
        ) / (self._model_params.Iz * self._gk_geometry.m)
        # Longitudinal Acceleration of kart
        acc_x = (f_x_front + f_x_rear) / self._gk_geometry.m + vrot_z * v_y
        # Lateral Acceleration of kart
        acc_y = (f_y_front + f_y_right_rear + f_y_left_rear) / self._gk_geometry.m - vrot_z * v_x

        rot_kart = geometry.rotation_matrix(yaw)
        gokart_vel = rot_kart @ jnp.array([v_x, v_y])

        x_dot = gokart_vel[0]
        y_dot = gokart_vel[1]
        yaw_dot = vrot_z
        yaw_rate_dot = rotacc_z
        vel_x_dot = acc_x
        vel_y_dot = acc_y
        # endregion
        return jnp.array([x_dot, y_dot, vel_x_dot, vel_y_dot, yaw_dot, yaw_rate_dot, acc_x, acc_y])

    def _get_reg_from_velocity(self, vx):
        vx_thresh = jnp.array([3.0])
        min_reg, max_reg = 0.1, 3
        reg_factor = jnp.max(vx_thresh - jnp.abs(vx), 0) / vx_thresh
        return min_reg + (max_reg - min_reg) * reg_factor

    def _mu_y_front(self, v_y, v_x, reg):
        return self._magic4p(self._sideslip(v_y=v_y, v_x=v_x, reg=reg), self._paj_params.front_paj)

    def _mu_y_rear(self, v_y, v_x, taccx, reg):
        mu_y = self._magic4p(self._sideslip(v_y=v_y, v_x=v_x, reg=reg), self._paj_params.rear_paj)
        return self._capfactor(taccx, self._paj_params.rear_paj.D * G) * mu_y

    def _sideslip(self, v_y, v_x, reg):
        return jnp.atan2(v_y, jnp.fabs(v_x) + reg)  # (1 / capfactor(taccx, D2=D2)) *)

    def _magic4p(self, slip, paj_params):
        return (
            paj_params.D
            * jnp.sin(
                paj_params.C
                * jnp.atan(
                    paj_params.B * slip
                    - paj_params.E * (paj_params.B * slip - paj_params.E * jnp.atan(paj_params.B * slip))
                )
            )[0]
        )

    def _capfactor(self, taccx: float, D: float):
        return jnp.sqrt(1 - self._satfun(jnp.pow(taccx / D, 2)))

    def _satfun(self, x: float):
        # special conditional operation for jax
        l = 0.8
        r = 1 - l

        def branch1(x):
            return x

        def branch2(x):
            d = (1 + r - x) / r
            return 1 - 0.25 * r * d * d

        def branch3(x):
            return 1.0

        y = jax.lax.cond(x < l, branch1, lambda x: jax.lax.cond(x < 1 + r, branch2, branch3, x), x)
        return y * 0.95


class IgnitionTricycleModel(TricycleModelABC):

    def __init__(
        self,
        gk_geometry: GoKartGeometry,
        model_params: TricycleParams,
        paj_params: PajieckaParams,
        dt: float = 0.1,
        normalize_actions: bool = True,
    ):
        super().__init__(gk_geometry, model_params, paj_params, dt, normalize_actions)

    def _dynamics_model(self, action: jax.Array, state: jnp.ndarray):
        """
        Note: all dynamics are normalized w.r.t. the normal force
        hence the name *_acc instead of *_force
        Action: beta: steering wheel angle
                AB_L: left rear wheel acceleration [-1, 1]
                AB_R: right rear wheel acceleration [-1, 1]
                braking: braking force (not used)?
        State:  x: x position
                y: y position
                vel_x: x velocity
                vel_y: y velocity
                acc_x: x acceleration
                acc_y: y acceleration
                yaw: yaw angle
                yaw_rate: yaw rate
                yaw_acc: yaw acceleration
        """

        # beta, AB_L, AB_R = jnp.split(action_array, 3, axis=-1)
        beta, AB_L, AB_R = action

        x, y, v_x, v_y, yaw, vrot_z, a_x, a_y = state  # float shape ()

        delta_left = self._ackermann_mapping_left(beta)
        delta_right = self._ackermann_mapping_right(beta)
        delta = jnp.array([delta_left, delta_right])

        Fzt = self._vertical_forces_tire(a_x, a_y)

        Vxt, Vyt = self._wheel_velocities_tire(v_x, v_y, vrot_z, delta)

        alpha = self._wheel_slip_angles_tire(Vxt, Vyt, Fzt, self._dt)

        # forces longitudinal (x-dir.) - tire frame
        # Fxt = jnp.array([0, 0, 0.5 * self._gk_geometry.m * AB_L, 0.5 * self._gk_geometry.m * AB_R])
        Fxt = jnp.array([0, 0, self._gk_geometry.m * AB_L, self._gk_geometry.m * AB_R]) # <- changed from original to match the actual inputs

        # forces lateral (y-dir.) - tire frame
        Fyt = self._pacejka(alpha, Fzt, Fxt)

        # forces - vehicle frame (fl, fr, rl, rr) & drag
        # longitudianl forces (x-dir.)
        Fx = jnp.array(
            [
                Fxt[0] * jnp.cos(delta[0]) - Fyt[0] * jnp.sin(delta[0]),
                Fxt[1] * jnp.cos(delta[1]) - Fyt[1] * jnp.sin(delta[1]),
                Fxt[2],
                Fxt[3],
            ]
        )
        # lateral forces (y-dir.)
        Fy = jnp.array(
            [
                Fxt[0] * jnp.sin(delta[0]) + Fyt[0] * jnp.cos(delta[0]),
                Fxt[1] * jnp.sin(delta[1]) + Fyt[1] * jnp.cos(delta[1]),
                Fyt[2],
                Fyt[3],
            ]
        )
        # vertical forces (z-dir.)
        Fz = jnp.array([Fzt[0], Fzt[1], Fzt[2], Fzt[3]])

        # drag force (x-dir.)
        Fd = 0.5 * v_x * self._gk_geometry.drag_a * self._gk_geometry.drag_c * RHO**2

        # Derivatives
        lf = self._gk_geometry.l1
        lr = self._gk_geometry.l2
        m = self._gk_geometry.m
        Jzz = self._model_params.Iz * m
        tf = self._gk_geometry.w1
        tr = self._gk_geometry.w2

        # velocity
        vel_x_dot = (1 / m) * (Fx[0] + Fx[1] + Fx[2] + Fx[3] - Fd) + v_y * vrot_z
        vel_y_dot = (1 / m) * (Fy[0] + Fy[1] + Fy[2] + Fy[3]) - v_x * vrot_z
        yaw_rate_dot = (1 / Jzz) * (
            lf * (Fy[0] + Fy[1]) - lr * (Fy[2] + Fy[3]) + (tf / 2) * (Fx[1] - Fx[0]) + (tr / 2) * (Fx[3] - Fx[2])
        )

        # acceleration
        a_x = vel_x_dot
        a_y = vel_y_dot

        # pose
        x_dot = jnp.cos(yaw) * v_x - jnp.sin(yaw) * v_y
        y_dot = jnp.sin(yaw) * v_x + jnp.cos(yaw) * v_y
        yaw_dot = vrot_z

        # endregion
        return jnp.array([x_dot, y_dot, vel_x_dot, vel_y_dot, yaw_dot, yaw_rate_dot, a_x, a_y])

    def _vertical_forces_tire(self, ax, ay):

        lr = self._gk_geometry.l2
        lf = self._gk_geometry.l1
        h = self._gk_geometry.h
        l = self._gk_geometry.l
        tf = self._gk_geometry.w1
        tr = self._gk_geometry.w2
        m = self._gk_geometry.m

        # vertical forces considering load transfers - (fl, fr, rl, rr)
        Fz_fl = -(lr * m * G) / (2 * l) + (h * m * ax) / (2 * l) + (h * m * ay) / tf
        Fz_fr = -(lr * m * G) / (2 * l) + (h * m * ax) / (2 * l) - (h * m * ay) / tf
        Fz_rl = -(lf * m * G) / (2 * l) - (h * m * ax) / (2 * l) + (h * m * ay) / tr
        Fz_rr = -(lf * m * G) / (2 * l) - (h * m * ax) / (2 * l) - (h * m * ay) / tr

        Fz = jnp.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr])

        return Fz

    def _wheel_velocities_tire(self, vx, vy, vtheta, delta):

        lf = self._gk_geometry.l1
        lr = self._gk_geometry.l2
        tf = self._gk_geometry.w1
        tr = self._gk_geometry.w2

        # wheel velocities in tire frame
        # front left   - fl
        Vx_fl = (vx - vtheta * (tf / 2)) * jnp.cos(delta[0]) + (vy + vtheta * lf) * jnp.sin(delta[0])
        Vy_fl = (-vx + vtheta * (tf / 2)) * jnp.sin(delta[0]) + (vy + vtheta * lf) * jnp.cos(delta[0])
        # front right  - fr
        Vx_fr = (vx + vtheta * (tf / 2)) * jnp.cos(delta[1]) + (vy + vtheta * lf) * jnp.sin(delta[1])
        Vy_fr = (-vx - vtheta * (tf / 2)) * jnp.sin(delta[1]) + (vy + vtheta * lf) * jnp.cos(delta[1])
        # rear left    - rl
        Vx_rl = vx - vtheta * (tr / 2)
        Vy_rl = vy - vtheta * lr
        # rear right   - rr
        Vx_rr = vx + vtheta * (tr / 2)
        Vy_rr = vy - vtheta * lr

        Vx = jnp.array([Vx_fl, Vx_fr, Vx_rl, Vx_rr], dtype=jnp.float32)
        Vy = jnp.array([Vy_fl, Vy_fr, Vy_rl, Vy_rr], dtype=jnp.float32)

        return Vx, Vy

    def _wheel_slip_angles_tire(self, Vx, Vy, Fz, dt):

        paj_front_lateral = PacejkaParamsIgnition.front_lateral
        paj_rear_lateral = PacejkaParamsIgnition.rear_lateral

        D1y_f = paj_front_lateral.D1y
        D2y_f = paj_front_lateral.D2y
        By_f = paj_front_lateral.By
        Cy_f = paj_front_lateral.Cy
        D1y_r = paj_rear_lateral.D1y
        D2y_r = paj_rear_lateral.D2y
        By_r = paj_rear_lateral.By
        Cy_r = paj_rear_lateral.Cy

        nu: float = 1.1  # numerical stability in slip computation unitless

        # front tyres - left & right
        Dy_fl = (D1y_f * Fz[0] + D2y_f) * Fz[0]
        Cy_fl = By_f * Cy_f * Dy_fl
        um_fl = (dt / 2) * Cy_fl / (Fz[0] / G)
        alpha_fl = jnp.atan2(Vy[0], jnp.where(Vx[0] > nu * um_fl, Vx[0], nu * um_fl))

        Dy_fr = (D1y_f * Fz[1] + D2y_f) * Fz[1]
        Cy_fr = By_f * Cy_f * Dy_fr
        um_fr = (dt / 2) * Cy_fr / (Fz[1] / G)
        alpha_fr = jnp.atan2(Vy[1], jnp.where(Vx[1] > nu * um_fr, Vx[1], nu * um_fr))

        # rear tyres - left & right
        Dy_rl = (D1y_r * Fz[2] + D2y_r) * Fz[2]
        Cy_rl = By_r * Cy_r * Dy_rl
        um_rl = (dt / 2) * Cy_rl / (Fz[2] / G)
        alpha_rl = jnp.atan2(Vy[2], jnp.where(Vx[2] > nu * um_rl, Vx[2], nu * um_rl))

        Dy_rr = (D1y_r * Fz[3] + D2y_r) * Fz[3]
        Cy_rr = By_r * Cy_r * Dy_rr
        um_rr = (dt / 2) * Cy_rr / (Fz[3] / G)
        alpha_rr = jnp.atan2(Vy[3], jnp.where(Vx[3] > nu * um_rr, Vx[3], nu * um_rr))

        alpha = jnp.array([alpha_fl, alpha_fr, alpha_rl, alpha_rr])

        return alpha

    def _pacejka(self, alpha, Fz, Fx):

        paj_front_lateral = PacejkaParamsIgnition.front_lateral
        paj_rear_lateral = PacejkaParamsIgnition.rear_lateral

        D1y_f = paj_front_lateral.D1y
        D2y_f = paj_front_lateral.D2y
        By_f = paj_front_lateral.By
        Cy_f = paj_front_lateral.Cy
        Ey_f = paj_front_lateral.Ey
        D1y_r = paj_rear_lateral.D1y
        D2y_r = paj_rear_lateral.D2y
        By_r = paj_rear_lateral.By
        Cy_r = paj_rear_lateral.Cy
        Ey_r = paj_rear_lateral.Ey
        ByG_r = paj_rear_lateral.ByG
        CyG_r = paj_rear_lateral.CyG

        # front tyres - left & right
        Dy_fl = (D1y_f * Fz[0] + D2y_f) * Fz[0]
        Fy0_fl = Dy_fl * jnp.sin(
            Cy_f * jnp.atan(By_f * alpha[0] - Ey_f * (By_f * alpha[0] - jnp.atan(By_f * alpha[0])))
        )
        G_fl = 1.0
        Fy_fl = G_fl * Fy0_fl

        Dy_fr = (D1y_f * Fz[1] + D2y_f) * Fz[1]
        Fy0_fr = Dy_fr * jnp.sin(
            Cy_f * jnp.atan(By_f * alpha[1] - Ey_f * (By_f * alpha[1] - jnp.atan(By_f * alpha[1])))
        )
        G_fr = 1.0
        Fy_fr = G_fr * Fy0_fr

        # rear tyres - left & right
        Dy_rl = (D1y_r * Fz[2] + D2y_r) * Fz[2]
        Fy0_rl = Dy_rl * jnp.sin(
            Cy_r * jnp.atan(By_r * alpha[2] - Ey_r * (By_r * alpha[2] - jnp.atan(By_r * alpha[2])))
        )
        mu_rl = jnp.abs(Fx[2]) / jnp.abs(Fz[2])
        G_rl = jnp.cos(CyG_r * jnp.atan(ByG_r * mu_rl))
        Fy_rl = G_rl * Fy0_rl

        Dy_rr = (D1y_r * Fz[3] + D2y_r) * Fz[3]
        Fy0_rr = Dy_rr * jnp.sin(
            Cy_r * jnp.atan(By_r * alpha[3] - Ey_r * (By_r * alpha[3] - jnp.atan(By_r * alpha[3])))
        )
        mu_rr = jnp.abs(Fx[3]) / jnp.abs(Fz[3])
        G_rr = jnp.cos(CyG_r * jnp.atan(ByG_r * mu_rr))
        Fy_rr = G_rr * Fy0_rr

        Fy = jnp.array([Fy_fl, Fy_fr, Fy_rl, Fy_rr])

        return Fy


class OriginalTricycleModel(TricycleModelABC):

    def __init__(
        self,
        gk_geometry: GoKartGeometry,
        model_params: TricycleParams,
        paj_params: PajieckaParams,
        dt: float = 0.1,
        normalize_actions: bool = True,
    ):
        super().__init__(gk_geometry, model_params, paj_params, dt, normalize_actions)

    def _dynamics_model(
        self,
        action: jax.Array,
        state: jnp.ndarray,
    ):
        """
        Note: all dynamics are normalized w.r.t. the normal force
        hence the name *_acc instead of *_force
        Action: beta: steering wheel angle
                AB_L: left rear wheel acceleration [-1, 1]
                AB_R: right rear wheel acceleration [-1, 1]
                braking: braking force (not used)?
        State:  x: x position
                y: y position
                vel_x: x velocity
                vel_y: y velocity
                yaw: yaw angle
                yaw_rate: yaw rate
        """

        # beta, AB_L, AB_R = jnp.split(action_array, 3, axis=-1)
        beta, AB_L, AB_R = action

        x, y, vel_x, vel_y, yaw, yaw_rate, a_x, a_y = state  # float shape ()

        # region front wheel acc
        # front wheel angle
        delta = self._ackermann_mapping(beta)  # steering angle

        # velocities at front wheel (Marc Heim, (2.43))
        vel1 = jnp.array([vel_x, vel_y + self._gk_geometry.l1 * yaw_rate])  # go kart frame
        delta_rotation = geometry.rotation_matrix(delta)

        # Adaption from Marc Heim (2.82f)
        v1_tyre = delta_rotation.T @ vel1  # front tyre frame
        # forces at front wheel, only lateral force, no longitudinal force
        acc_f1y = self._get_front_acc_y(v1_tyre[1], v1_tyre[0])
        delta_rotation_reverse = geometry.rotation_matrix(-delta)
        F1 = delta_rotation_reverse.T @ jnp.array([0.0, acc_f1y])
        # Marc Heim (2.82f) front acc in go kart frame
        # endregion

        # region calculate back axle acc
        total_acc = AB_L + AB_R
        # lat velocity at back axle, vx doesn't change Marc Heim (2.43)
        v2y = vel_y - self._gk_geometry.l2 * yaw_rate  #  Linearized?  go kart frame
        F2_n = self._gk_geometry.F2n
        # Lateral acceleration from from left rear wheel Marc Heim (2.77)
        # F2l_y = self._get_rear_acc_y(v2y, vel_x, (AB_L / 2) / F2_n) * F2_n / 2
        
        # F2l_y = self._get_rear_acc_y(v2y, vel_x, AB_L / 2) * F2_n / 2
        F2l_y = self._get_rear_acc_y(v2y, vel_x, AB_L) * F2_n / 2  # <- changed from original to match the actual inputs
        
        # Lateral acceleration from from right rear wheel Marc Heim (2.77)
        # F2r_y = self._get_rear_acc_y(v2y, vel_x, (AB_R / 2) / F2_n) * F2_n / 2
        
        # F2r_y = self._get_rear_acc_y(v2y, vel_x, AB_R / 2) * F2_n / 2
        F2r_y = self._get_rear_acc_y(v2y, vel_x, AB_R) * F2_n / 2  # <- changed from original to match the actual inputs
        
        # Lateral acceleration from rear wheels
        F2y = self._get_rear_acc_y(v2y, vel_x, total_acc / F2_n) * F2_n
        # endregion

        # region Torque from difference in real wheel accelerations (Marc Heim. 2.79)

        cog2rearwheel = jnp.sqrt(
            self._gk_geometry.l2 * self._gk_geometry.l2 + (self._gk_geometry.w2 / 2) * (self._gk_geometry.w2 / 2)
        )
        tv2orthogonal = jnp.atan2(self._gk_geometry.l2, self._gk_geometry.w2 / 2)
        lever = cog2rearwheel * tv2orthogonal * 2

        # tv_trq = 0.5 * (AB_R - AB_L) * lever
        tv_trq = (AB_R - AB_L) * lever  # <- changed from original to match the actual inputs
        # endregion

        # region Gokart accelerations
        # Rotational Acceleration of the kart Marc Heim (2.88, 2.91)
        # rotacc_z = (tv_trq + F1[1] * self.gk_geometry.l1 - F2y * self.gk_geometry.l2) / self.model_params.Iz
        rotacc_z = (
            tv_trq + F1[1] * self._gk_geometry.l1 - (F2l_y + F2r_y) * self._gk_geometry.l2
        ) / self._model_params.Iz
        # Forward Acceleration of kart Marc Heim (2.86, 2.89), extended
        acc_x = F1[0] + total_acc + yaw_rate * vel_y
        # Lateral Acceleration of kart Marc Heim (2.87, 2.90)
        acc_y = F1[1] + F2l_y + F2r_y - yaw_rate * vel_x

        rot_kart = geometry.rotation_matrix(yaw)
        lv = jnp.array([vel_x, vel_y])
        gokart_vel = rot_kart @ lv
        # endregion

        # region prepare output vector

        x_dot = gokart_vel[0]
        y_dot = gokart_vel[1]
        yaw_dot = yaw_rate
        yaw_rate_dot = rotacc_z
        vel_x_dot = acc_x
        vel_y_dot = acc_y
        # endregion
        return jnp.array(
            [
                x_dot,
                y_dot,
                vel_x_dot,
                vel_y_dot,
                yaw_dot,
                yaw_rate_dot,
                acc_x,
                acc_y,
            ]
        )

    def _ackermann_mapping(self, steering: float) -> float:
        """Maps angle of steerig wheel to the steering angle of front wheel."""
        return -0.065 * steering * steering * steering + 0.45 * steering

    def _get_front_acc_y(self, v_y: float, v_x: float):
        return self._magic(-v_y / (v_x + self._model_params.REG_), self._paj_params.front_paj)

    def _get_rear_acc_y(self, v_y, v_x, taccx):
        # taccx equals to f*x according to M.H.
        s = self._simpleslip(v_y, v_x, taccx, self._paj_params.rear_paj.D)
        acc_y = self._magic(s, self._paj_params.rear_paj)
        return self._capfactor(taccx, self._paj_params.rear_paj.D) * acc_y

    def _magic(self, slipping_coef, paj_params):
        return paj_params.D * jnp.sin(paj_params.C * jnp.atan(paj_params.B * slipping_coef))

    def _simpleslip(self, v_y: float, v_x: float, taccx: float, D: float):
        return -(1 / self._capfactor(taccx, D)) * v_y / (v_x + self._model_params.REG_)

    def _capfactor(self, taccx: float, D: float):
        return jnp.sqrt(1 - self._satfun(jnp.pow(taccx / D, 2)))

    # def _satfun(self, x: float):
    #   l = 0.8
    #   r = 1 - l
    #   if x < l:
    #     y = x
    #   else:
    #     if x < 1 + r:
    #       d = (1 + r - x) / r
    #       y = 1 - 1.0 / 4 * r * d * d
    #     else:
    #       y = 1
    #   y *= 0.95
    #   return y

    def _satfun(self, x: float):
        # special conditional operation for jax
        l = 0.8
        r = 1 - l

        def branch1(x):
            return x

        def branch2(x):
            d = (1 + r - x) / r
            return 1 - 0.25 * r * d * d

        def branch3(x):
            return 1.0

        y = jax.lax.cond(x < l, branch1, lambda x: jax.lax.cond(x < 1 + r, branch2, branch3, x), x)
        return y * 0.95
