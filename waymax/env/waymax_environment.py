from typing import Tuple
import copy
import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.env import PlanningAgentEnvironment, PlanningAgentSimulatorState
from waymax.datatypes.observation import sdc_observation_from_state

class WaymaxDrivingEnvironment(PlanningAgentEnvironment):
    """
    The WaymaxDrivingEnvironment inherits from the PlanningAgentEnvironment
    to write our own observation function and override the reset function and
    the step function to be consisitent with the GokartRacingEnvironment.
    """

    def observe(self, state: PlanningAgentSimulatorState) -> jax.Array:
        transformed_obs = sdc_observation_from_state(state)

        other_objects_xy = jnp.squeeze(transformed_obs.trajectory.xy).reshape(-1)
        flattened_mask = transformed_obs.is_ego.reshape(-1)
        indices = jnp.where(flattened_mask>0, jnp.arange(len(flattened_mask)), -1)
        indices = jnp.sort(indices)
        index = indices[-1]
        rg_xy = jnp.squeeze(transformed_obs.roadgraph_static_points.xy).reshape(-1)
        sdc_vel_xy = jnp.squeeze(transformed_obs.trajectory.vel_xy)[index,:].reshape(-1)

        #TODO: (tian) to delete the zeros in other_objects_xy
        obs = jnp.concatenate(
                [other_objects_xy, rg_xy, sdc_vel_xy,],
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
        obs = self.observe(new_state)
        done = new_state.is_done
        obs, new_state = jax.lax.cond(
            done,
            lambda _: self.reset(new_state),
            lambda _: (obs, new_state),
            operand=None
        )
        info ={}

        return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(new_state), reward, done, info