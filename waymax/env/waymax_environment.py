from typing import Tuple
import copy
import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.env import PlanningAgentEnvironment, PlanningAgentSimulatorState
from waymax.datatypes.observation import sdc_observation_from_state
from waymax.utils import geometry

class WaymaxDrivingEnvironment(PlanningAgentEnvironment):
    """
    The WaymaxDrivingEnvironment inherits from the PlanningAgentEnvironment
    to write our own observation function and override the reset function and
    the step function to be consisitent with the GokartRacingEnvironment.
    """

    def observe(self, state: PlanningAgentSimulatorState) -> jax.Array:
        transformed_obs, pose = sdc_observation_from_state(state, roadgraph_top_k=100, verbose=True)

        other_objects_xy = jnp.squeeze(transformed_obs.trajectory.xy).reshape(-1)
        flattened_mask = transformed_obs.is_ego.reshape(-1)
        indices = jnp.where(flattened_mask>0, jnp.arange(len(flattened_mask)), -1)
        indices = jnp.sort(indices)
        index = indices[-1]
        rg_xy = jnp.squeeze(transformed_obs.roadgraph_static_points.xy).reshape(-1)
        sdc_vel_xy = jnp.squeeze(transformed_obs.trajectory.vel_xy)[index,:].reshape(-1)
        # global_tar_1 = state.log_trajectory.xy[index, state.timestep+5].reshape(-1,2)
        # tar_1 = geometry.transform_points(pts=global_tar_1, pose_matrix=pose.matrix,).reshape(-1)
        # global_tar_2 = state.log_trajectory.xy[index, state.timestep+10].reshape(-1,2)
        # tar_2 = geometry.transform_points(pts=global_tar_2, pose_matrix=pose.matrix,).reshape(-1)
        global_tar = jnp.where(state.timestep>=45, state.log_trajectory.xy[index, -1].reshape(-1,2), state.log_trajectory.xy[index, 45].reshape(-1,2))
        tar_1 = geometry.transform_points(pts=global_tar, pose_matrix=pose.matrix,).reshape(-1)

        #TODO: (tian) to delete the zeros in other_objects_xy
        obs = jnp.concatenate(
                [other_objects_xy, rg_xy, tar_1, sdc_vel_xy,],
                axis=-1)
        return obs

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array | None = None) -> Tuple[jax.Array, PlanningAgentSimulatorState]:
        state = super().reset(state, rng)
        obs = self.observe(state)

        return obs, state
    
    def step(
            self, state: PlanningAgentSimulatorState, action: datatypes.Action, rng: jax.Array | None = None
    ) -> Tuple[jax.Array, PlanningAgentSimulatorState, jax.Array, bool, ]:
        last_state = copy.deepcopy(state)
        new_state = super().step(last_state, action, rng)
        reward = super().reward(last_state, action)
        metrics = super().metrics(last_state)
        # TODO: (tian)
        reward_dict = {
            "progression_reward": metrics['log_divergence'].value,
            "orientation_reward": metrics['overlap'].value,
            "offroad_reward": metrics['offroad'].value
        }
        obs = self.observe(new_state)
        done = new_state.is_done
        # done = jnp.logical_or(new_state.is_done, metrics['overlap'].value==1)
        # done = jnp.logical_or(done, metrics['offroad'].value==1)
        obs, new_state = jax.lax.cond(
            done,
            lambda _: self.reset(new_state),
            lambda _: (obs, new_state),
            operand=None
        )
        info = reward_dict

        return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(new_state), reward, done, info