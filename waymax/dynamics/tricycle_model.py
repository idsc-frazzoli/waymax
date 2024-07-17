# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the bicycle (acceleration, steering) dynamics model.



This action space always uses the [-1.0, 1.0] as the range for acceleration
and steering commands to be consistent with other RL training pipeline since
many algorithms' hyperparameters are tuned based on this assumption. The actual
acceleration and steering command range can still be specified by `max_accel`
and `max_steering` in the class definition function.
"""

from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np

from waymax import datatypes
from waymax.dynamics import abstract_dynamics
from waymax.utils import geometry
# from waymax.config import GoKartGeometry, TricycleParams, PajieckaParams

DynamicsModel = abstract_dynamics.DynamicsModel
# TODO Determine whether 0.6 is appropriate speed limit.
# This speed limit helps to filter out false positive very large steering value.
_SPEED_LIMIT = 0.6  # Units: m/s

class GoKartGeometry:
  l1: float = 0.5 # Distance from cog to front tires
  l2: float = 0.5 # Distance from cog to rear tires
  w1: float = 1 # Distance between front Tires
  w2: float = 1 # Distance between rear Tires
  back2backaxle: float = 0.3 # Distance from the rear of the gokart to the back axle
  frontaxle2front: float = 0.3 # Distance from the front axle to the front of the kart
  wheel2border: float = 0.2 # Side distance between center of the wheel and external frame
  F2n: float = l1/(l1 + l2) # Normal force at the rear axle

class TricycleParams:
  Iz: float = 0.7 # Inertia around the z axis
  REG_: float = 0.5 # Regularization factor
  
class PajieckaParams:
  class front_paj:
    B: float = 13.17
    C: float = 1.26
    D: float = 0.8
    E: float = 0.42
  class rear_paj:
    B: float = 9.02
    C: float = 1.27
    D: float = 0.97
    E: float = 0.21

class TricycleModel(DynamicsModel):
  """Dynamics model using acceleration and steering curvature for control."""

  def __init__(
      self,
      gk_geometry: GoKartGeometry,
      model_params: TricycleParams,
      paj_params: PajieckaParams,
      dt: float = 0.1,
      max_accel: float = 6.0,
      max_steering: float = 0.3,
      normalize_actions: bool = False,
  ):
    """Initializes the bounds of the action space.

    Args:
      dt: The time length per step used in the simulator in seconds.
      max_accel: The maximum acceleration magnitude.
      max_steering: The maximum steering curvature magnitude, which is the
        inverse of the turning radius (the minimum radius of available space
        required for that vehicle to make a circular turn).
      normalize_actions: Whether to normalize the action range to [-1,1] or not.
        By default it uses the unnormalized range and in order to train with RL,
        such as with ACME. Ideally we should normalize the ranges.
    """
    super().__init__()
    self.gk_geometry = gk_geometry
    self.model_params = model_params
    self.paj_params = paj_params
    self._dt = dt
    self._max_accel = max_accel
    self._max_steering = max_steering
    self._normalize_actions = normalize_actions

  def action_spec(self) -> specs.BoundedArray:
    """Action spec for the acceleration steering continuous action space."""
    if not self._normalize_actions:
      return specs.BoundedArray(
          # last dim: (acceleration, steering)
          shape=(3,),
          dtype=np.float32,
          minimum=np.array([-self._max_steering, -self._max_accel, -self._max_accel]),
          maximum=np.array([self._max_steering, self._max_accel, self._max_accel]),
      )
    else:
      return specs.BoundedArray(
          # last dim: (acceleration, steering)
          shape=(3,),
          dtype=np.float32,
          minimum=np.array([-1.0, -1.0, -1.0]),
          maximum=np.array([1.0, 1.0, 1.0]),
      )

  def _clip_values(self, action_array: jax.Array) -> jax.Array:
    """Clip action values to be within the allowable ranges."""
    
    steering = jnp.clip(
        action_array[..., 0],
        self.action_spec().minimum[0],
        self.action_spec().maximum[0],
    )
    acc_l = jnp.clip(
        action_array[..., 1],
        self.action_spec().minimum[1],
        self.action_spec().maximum[1],
    )
    acc_r = jnp.clip(
        action_array[..., 2],
        self.action_spec().minimum[2],
        self.action_spec().maximum[2],
    )
    return jnp.stack([steering, acc_l, acc_r], axis=-1)

  @jax.named_scope('InvertibleBicycleModel.compute_update')
  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.GoKartTrajectory,
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

    # Vectorize _RK4_update function along batch and num_objects dimensions
    if len(x.shape) == 2:  # x shape (num_objects, num_timesteps=1)
      rk4_vmap = jax.vmap(self._RK4_update, in_axes=(0, 0, None))
    elif len(x.shape) == 3: # x shape (batch_size, num_objects, num_timesteps=1)
      rk4_vmap = jax.vmap(jax.vmap(self._RK4_update, in_axes=(0, 0, None)), in_axes=(0, 0, None))

    new_states = rk4_vmap(action.data, state, t)
    
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
    '''
    Note: all dynamics are normalized w.r.t. the normal force
    hence the name *_acc instead of *_force
    Action: beta: steering wheel angle
            AB_L: left rear wheel acceleration [-1, 1]
            AB_R: right rear wheel acceleration [-1, 1]
            braking: braking force (not used)?
    State:  X: x position
            Y: y position
            Vx: x velocity
            Vy: y velocity
            yaw: yaw angle
            yaw_rate: yaw rate
    '''

    action_array = self._clip_values(action)

    # beta, AB_L, AB_R = jnp.split(action_array, 3, axis=-1)
    beta, AB_L, AB_R = action_array

    # beta = action.beta
    # AB_L = action.AB_L
    # AB_R = action.AB_R

    # x = state[0]
    # y = state[1]
    # vel_x = state[2]
    # vel_y = state[3]
    # yaw = state[4]
    # # yaw_rate = trajectory.yaw_rate
    # yaw_rate = state[5]
    x, y, vel_x, vel_y, yaw, yaw_rate = state # float shape ()



    #region front wheel acc
    # front wheel angle
    delta = self._ackermann_mapping(beta) # steering angle

    # velocities at front wheel (Marc Heim, (2.43))
    vel1 = jnp.array([vel_x, vel_y + self.gk_geometry.l1 * yaw_rate])   # go kart frame
    delta_rotation = self._rotation_matrix(delta)

    # Adaption from Marc Heim (2.82f)
    v1_tyre = delta_rotation.T @ vel1   # front tyre frame
    # forces at front wheel, only lateral force, no longitudinal force
    acc_f1y = self._get_front_acc_y(v1_tyre[1], v1_tyre[0])
    delta_rotation_reverse = self._rotation_matrix(-delta)
    F1 = delta_rotation_reverse.T @ jnp.array([0.0, acc_f1y]); # Marc Heim (2.82f) front acc in go kart frame
    # endregion

    # region calculate back axle acc
    total_acc = AB_L + AB_R
    # lat velocity at back axle, vx doesn't change Marc Heim (2.43)
    v2y = vel_y - self.gk_geometry.l2 * yaw_rate  #  Linearized?  go kart frame
    F2_n = self.gk_geometry.F2n
    # Lateral acceleration from from left rear wheel Marc Heim (2.77)
    F2l_y = self._get_rear_acc_y(v2y, vel_x, (AB_L / 2) / F2_n) * F2_n / 2
    # Lateral acceleration from from right rear wheel Marc Heim (2.77)
    F2r_y = self._get_rear_acc_y(v2y, vel_x, (AB_R / 2) / F2_n) * F2_n / 2 
    # Lateral acceleration from rear wheels
    F2y = self._get_rear_acc_y(v2y, vel_x, total_acc / F2_n) * F2_n
    # endregion

    # region Torque from difference in real wheel accelerations (Marc Heim. 2.79)

    cog2rearwheel = jnp.sqrt(self.gk_geometry.l2 * self.gk_geometry.l2 + (self.gk_geometry.w2 / 2) *(self.gk_geometry.w2 / 2))
    tv2orthogonal = jnp.atan2(self.gk_geometry.l2, self.gk_geometry.w2 / 2)
    lever = cog2rearwheel * tv2orthogonal * 2

    tv_trq = .5 * (AB_R - AB_L) * lever
    #endregion

    #region Gokart accelerations
    # Rotational Acceleration of the kart Marc Heim (2.88, 2.91)
    rotacc_z = (tv_trq + F1[1] * self.gk_geometry.l1 - F2y * self.gk_geometry.l2) / self.model_params.Iz
    # Forward Acceleration of kart Marc Heim (2.86, 2.89), extended
    acc_x = F1[0] + total_acc + yaw_rate * vel_y
    # Lateral Acceleration of kart Marc Heim (2.87, 2.90)
    acc_y = F1[1] + F2l_y + F2r_y - yaw_rate * vel_x

    rot_kart = self._rotation_matrix(yaw)
    lv = jnp.array([vel_x, vel_y])
    gokart_vel = rot_kart @ lv
    #endregion

    # region prepare output vector
    
    x_dot = gokart_vel[0]
    y_dot = gokart_vel[1]
    yaw_dot = yaw_rate
    yaw_rate_dot = rotacc_z
    vel_x_dot = acc_x
    vel_y_dot = acc_y
    # endregion
    return jnp.array([x_dot, y_dot, vel_x_dot, vel_y_dot, yaw_dot, yaw_rate_dot])

  def _ackermann_mapping(self, steering: float) -> float:
    """Maps angle of steerig wheel to the steering angle of front wheel."""
    return -0.065 * steering * steering * steering + 0.45 * steering

  def _rotation_matrix(self, theta):
    """
    Create a 2D rotation matrix for a given angle theta.
    """
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]
    ])

  def _get_front_acc_y(self, v_y: float, v_x: float):
    return self._magic(-v_y / (v_x + self.model_params.REG_), self.paj_params.front_paj)

  def _get_rear_acc_y(self, v_y,  v_x, taccx):
    # taccx equals to f*x according to M.H.
    s = self._simpleslip(v_y, v_x, taccx, self.paj_params.rear_paj.D)
    acc_y = self._magic(s, self.paj_params.rear_paj)
    return self._capfactor(taccx, self.paj_params.rear_paj.D) * acc_y


  def _magic(self, slipping_coef, paj_params):
    return paj_params.D * jnp.sin(paj_params.C * jnp.atan(paj_params.B * slipping_coef))

  def _simpleslip(self, v_y: float, v_x: float , taccx: float, D: float):
    return -(1 / self._capfactor(taccx, D)) * v_y / (v_x + self.model_params.REG_)

  def _capfactor(self, taccx: float, D: float) :
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

  def _RK4_update(self, action, state, dt):
    '''
    Runge-Kutta 4th order integration
    '''
    k1 = self._dynamics(action, state)
    k2 = self._dynamics(action, state + dt / 2 * k1)
    k3 = self._dynamics(action, state + dt / 2 * k2)
    k4 = self._dynamics(action, state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
  
  def _euler_forward(self, state, action, dt):
    '''
    Euler forward integration
    '''
    return state + dt * self._dynamics(state, action)
  
  def inverse():
    pass

if __name__ == '__main__':
  # from waymax.config import GoKartGeometry, TricycleParams, PajieckaParams
  from waymax.datatypes import Action, Trajectory

  gk_geometry = GoKartGeometry()
  model_params = TricycleParams()
  paj_params = PajieckaParams()
  tricycle = TricycleModel(gk_geometry, model_params, paj_params)
  action = Action(data=jnp.array([0.1, 0.1, 0.1]), valid=jnp.array([True, True, True]))
  trajectory = Trajectory.zeros((1,))
  print(tricycle.compute_update(action, trajectory))
 
  