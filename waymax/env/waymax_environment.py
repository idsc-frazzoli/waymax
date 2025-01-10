import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.env import PlanningAgentEnvironment, PlanningAgentSimulatorState
from waymax.datatypes.observation import sdc_observation_from_state
from waymax.utils import geometry
from dm_env.specs import BoundedArray

class WaymaxDrivingEnvironment(PlanningAgentEnvironment):
  """
  The WaymaxDrivingEnvironment inherits from the PlanningAgentEnvironment
  to implement the observe function and override other functions when necessary,
  meanwhile consisitent with the GokartRacingEnvironment.
  """

  def observe(self, state: PlanningAgentSimulatorState,  rng: jax.Array | None = None,) -> jax.Array:
    del rng
    
    transformed_obs, pose = sdc_observation_from_state(state, roadgraph_top_k=100, verbose=True)
    # 1. road information (relative poses of closest 100 edege points)
    rg_xy = jnp.squeeze(transformed_obs.roadgraph_static_points.xy).reshape(-1)
    # 2. own state (velocity_x, velocity_y in local frame)
    flattened_mask = transformed_obs.is_ego.reshape(-1)
    indices = jnp.where(flattened_mask>0, jnp.arange(len(flattened_mask)), -1)
    indices = jnp.sort(indices)
    index = indices[-1]
    sdc_speed = jnp.squeeze(transformed_obs.trajectory.vel_xy)[index,:].reshape(-1)
    # 3. others' state (relative poses and bbox dimensions)
    distances = jnp.linalg.norm(
      jnp.squeeze(transformed_obs.trajectory.xy), axis=-1
    )
    valid_distances = jnp.where(jnp.logical_and(jnp.squeeze(transformed_obs.trajectory.valid), jnp.logical_not(jnp.squeeze(transformed_obs.is_ego))), distances, 0.0)
    top_dist, _ = jax.lax.top_k(valid_distances, 1)
    other_objects_info_raw = jnp.squeeze(transformed_obs.trajectory.stack_fields(['x','y','yaw','length','width']))
    mask_values = [-top_dist, 0.0, 0.0]
    for attr in range(3):
        masked_attr = jnp.where(jnp.squeeze(transformed_obs.trajectory.valid), other_objects_info_raw[:,attr], mask_values[attr])
        other_objects_info_raw = other_objects_info_raw.at[:,attr].set(masked_attr)
    other_distances = jnp.where(jnp.logical_not(jnp.squeeze(transformed_obs.is_ego)), distances, float('inf'))
    _, other_idx = jax.lax.top_k(-other_distances, self.config.max_num_objects-1)
    other_idx = jnp.sort(other_idx)
    other_objects_info = (jnp.take_along_axis(other_objects_info_raw, other_idx[..., None], axis=-2)).reshape(-1)
    # 4. navigation information (relative poses of current reference point and next 5 reference points with stride 2)
    tars = []
    stride = 2
    horizon = 5
    for t_ele in range(1+horizon):
      global_xy = state.log_trajectory.xy[index, state.timestep+t_ele*stride].reshape(1,2)
      tars.append(geometry.transform_points(pts=global_xy, pose_matrix=pose.matrix).reshape(-1))
      # global_vel_xy = state.log_trajectory.vel_xy[index, state.timestep+t_ele*stride].reshape(1,2)
      # tars.append(geometry.transform_direction(pts_dir=global_vel_xy, pose_matrix=pose.matrix).reshape(-1))
      # global_yaw = state.log_trajectory.yaw[index, state.timestep+t_ele*stride].reshape(1,)
      # tars.append(((global_yaw + pose.delta_yaw + 2*jnp.pi) % (2*jnp.pi) - jnp.pi).reshape(1,))
    tars = jnp.concatenate(tars)

    obs = jnp.concatenate(
      [rg_xy, other_objects_info, tars, sdc_speed], axis=-1
    )
    return obs
  
  def observation_spec(self) -> BoundedArray:
    # TODO: (tian) find a proper place to assert obs_dim
    dim = 200 + 2 + 75 + 12
    minimum = -jnp.array([jnp.inf] * dim)
    maximum = jnp.array([jnp.inf] * dim)
    specs = BoundedArray((dim,), jnp.float32, minimum, maximum)
    return specs
  
  def reset(
    self, state: datatypes.SimulatorState, rng: jax.Array | None = None
  ) -> PlanningAgentSimulatorState:
    init_state: PlanningAgentSimulatorState = super().reset(state, rng)
    len_actions_history = self.config.len_actions_history
    init_actions_history = datatypes.SDC_actions_history(data=jnp.zeros(state.shape + (len_actions_history,) + self.action_spec().shape), valid=jnp.zeros((state.shape + (len_actions_history,1,)), dtype=jnp.bool_))
    init_actions_history = init_actions_history.init()
    return init_state.replace(actions_history=init_actions_history)

  def step(
    self, state: PlanningAgentSimulatorState, action: datatypes.Action, rng: jax.Array | None = None,
  ) -> PlanningAgentSimulatorState:
    new_state: PlanningAgentSimulatorState = super().step(state, action, rng)
    updated_actions_history = state.actions_history.update(action)
    return new_state.replace(actions_history=updated_actions_history)

  def action_spec(self) -> BoundedArray:
    data_spec = self.dynamics.action_spec()
    return data_spec
  
  def termination(self, state: PlanningAgentSimulatorState) -> jax.Array:
    metric_dict = self.metrics(state)
    is_offroad = metric_dict["offroad"].value.astype(jnp.bool)
    is_overlap = metric_dict["overlap"].value.astype(jnp.bool)
    condition = jnp.logical_or(is_offroad, is_overlap)
    condition = jnp.logical_or(condition, state.is_done)
    return condition.squeeze()
  
  def truncation(self, state: PlanningAgentSimulatorState) -> jax.Array:
    return (jnp.zeros(state.shape)).astype(jnp.bool_)
