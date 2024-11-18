"""
Implementation of the bicycle (acceleration, steering) dynamics model.

This action space always uses the [-1.0, 1.0] as the range for acceleration
and steering commands to be consistent with other RL training pipeline since
many algorithms' hyperparameters are tuned based on this assumption.
The actual acceleration and steering command range can still be specified by `max_accel`
and `max_steering` in the class definition function.
"""

import jax
import jax.numpy as jnp
import numpy as np
from dm_env import specs

from waymax import datatypes
from waymax.dynamics import abstract_dynamics
from waymax.utils import geometry
from waymax.utils.gokart_config import GoKartGeometry, TricycleParams, PajieckaParams

DynamicsModel = abstract_dynamics.DynamicsModel
_G = 9.81 # Units: m/s^2

class TricycleModel(DynamicsModel):
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
    #super().__init__()
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
                      [-self._model_params.max_steering, -self._model_params.max_accel, -self._model_params.max_accel]),
              maximum=np.array(
                      [self._model_params.max_steering, self._model_params.max_accel, self._model_params.max_accel]),
      )

  def _clip_values(self, action_array: jax.Array) -> jax.Array:
    """Clip action values to be within the allowable ranges."""
    return jnp.clip(
        action_array,
        jnp.asarray(self.action_spec().minimum),
        jnp.asarray(self.action_spec().maximum)
    )

  @jax.named_scope('TricycleModel.compute_update')
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

    x = trajectory.x   # shape (..., num_objects, num_timesteps=1)
    y = trajectory.y
    vel_x = trajectory.vel_x
    vel_y = trajectory.vel_y
    yaw = trajectory.yaw
    yaw_rate = trajectory.yaw_rate
    # yaw_rate = jnp.zeros_like(vel_x)
    state = jnp.concatenate((x, y, vel_x, vel_y, yaw, yaw_rate), axis=-1)

    action_clipped = self._clip_values(action.data)

    # Vectorize _RK4_update function along batch and num_objects dimensions
    if len(x.shape) == 2:  # x shape (num_objects, num_timesteps=1)
      rk4_vmap = jax.vmap(self._RK4_update, in_axes=(0, 0, None))
    elif len(x.shape) == 3: # x shape (batch_size, num_objects, num_timesteps=1)
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
        valid=trajectory.valid & action.valid,
    )
  def _dynamics(self, action: jax.Array, state: jnp.ndarray,):
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

    if self._normalize_actions:
      # scale back up if actions are normalized
      raw_act_min = -jnp.array([self._model_params.max_steering, self._model_params.max_accel, self._model_params.max_accel])
      raw_act_max = jnp.array([self._model_params.max_steering, self._model_params.max_accel, self._model_params.max_accel])
      act_min = self.action_spec().minimum
      act_max = self.action_spec().maximum
      # convert normalized action [-1,1] to a real world action (eg acceleration [-6,-6])
      action = raw_act_min + (raw_act_max - raw_act_min)*(action - act_min)/(act_max - act_min)

    # beta, AB_L, AB_R = jnp.split(action_array, 3, axis=-1)
    beta, AB_L, AB_R = action
    AB = AB_L + AB_R
    tv = (AB_R - AB_L)

    x, y, v_x, v_y, yaw, vrot_z = state # float shape ()

    reg = jnp.array([self._model_params.REG_]) # self._get_reg_from_velocity(v_x)

    #region front wheel acc
    # front wheel angle
    delta = self._ackermann_mapping(beta, two_wheels=False) # steering angle

    # velocities at front wheel (Marc Heim, (2.43))
    vel1 = jnp.array([v_x, v_y + self._gk_geometry.l1 * vrot_z])   # go kart frame
    # Tire Forces (velocity in wheels reference frame)
    rot_delta = geometry.rotation_matrix(delta)

    # # Front wheel velocity in wheel reference frame (Adaption from Marc Heim (2.82f))
    v_frontaxle = rot_delta.T @ vel1   # front tyre frame
    # friction coefficient front
    mu_front = self._mu_y_front(v_frontaxle[1], v_frontaxle[0], reg=reg)

    # longitudinal load transfer
    load_transfer = self._gk_geometry.h * AB
    fz_f = - (_G * self._gk_geometry.l2 - load_transfer) * self._gk_geometry.m / self._gk_geometry.l
    fz_r = - (_G * self._gk_geometry.l1 + load_transfer) * self._gk_geometry.m / self._gk_geometry.l

    # Longitudinal and lateral force in kart frame at the front wheel
    # jax.debug.print("shapes: mu_front = {}, fz_f = {}", mu_front.shape, fz_f.shape)
    # jax.debug.print("mu_front = {}, fz_f = {}\n", mu_front, fz_f)
    f_front_wheel = rot_delta @ jnp.array([0.0,  mu_front * fz_f])
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
    rotacc_z = (tv_torque + f_y_front * self._gk_geometry.l1 - (f_y_right_rear + f_y_left_rear) * self._gk_geometry.l2) / \
               (self._model_params.Iz * self._gk_geometry.m)
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
    return jnp.array([x_dot, y_dot, vel_x_dot, vel_y_dot, yaw_dot, yaw_rate_dot])

  def _ackermann_mapping(self, steering: float, two_wheels:bool = False) -> float:
    """Maps angle of steerig wheel to the steering angle of front wheel."""
    if two_wheels:
      return (self._ackermann_mapping_left(steering) + self._ackermann_mapping_right(steering))/2
    else:
      return -0.065 * steering * steering * steering + 0.45 * steering
      # return -0.04253 * steering * steering * steering + 4.455e-05 * steering * steering + 0.4039 * steering

  def _ackermann_mapping_left(self, steering: float) -> float:
    """Maps angle of steerig wheel to the steering angle of front left wheel."""
    return -0.0355 * steering * steering * steering - 0.0455 * steering * steering + 0.36 * steering

  def _ackermann_mapping_right(self, steering: float) -> float:
    """Maps angle of steerig wheel to the steering angle of front right wheel."""
    return -0.0355 * steering * steering * steering + 0.0455 * steering * steering + 0.36 * steering

  def _get_reg_from_velocity(self, vx):
    vx_thresh = jnp.array([3.0])
    min_reg, max_reg = 0.1, 3
    reg_factor = jnp.max(vx_thresh - jnp.abs(vx), 0) / vx_thresh
    return min_reg + (max_reg - min_reg) * reg_factor

  def _mu_y_front(self, v_y, v_x, reg):
    return self._magic4p(self._sideslip(v_y=v_y, v_x=v_x, reg=reg), self._paj_params.front_paj)

  def _mu_y_rear(self, v_y, v_x, taccx, reg):
    mu_y = self._magic4p(self._sideslip(v_y=v_y, v_x=v_x, reg=reg), self._paj_params.rear_paj)
    return self._capfactor(taccx, self._paj_params.rear_paj.D * _G) * mu_y

  def _sideslip(self, v_y, v_x, reg):
    return jnp.atan2(v_y, jnp.fabs(v_x) + reg)  # (1 / capfactor(taccx, D2=D2)) *)

  def _magic4p(self, slip, paj_params):
    return paj_params.D * jnp.sin(
      paj_params.C * jnp.atan(
        paj_params.B * slip - paj_params.E * (
          paj_params.B * slip - paj_params.E * jnp.atan(paj_params.B * slip)
        )
      )
    )[0]

  def _capfactor(self, taccx: float, D: float) :
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

  def _RK4_update(self, action, state, dt):
    """
    Runge-Kutta 4th order integration
    """
    k1 = self._dynamics(action, state)
    k2 = self._dynamics(action, state + dt / 2 * k1)
    k3 = self._dynamics(action, state + dt / 2 * k2)
    k4 = self._dynamics(action, state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  def _euler_forward(self, state, action, dt):
    """
    Euler forward integration
    """
    return state + dt * self._dynamics(state, action)

  def inverse(self):
    raise NotImplementedError



