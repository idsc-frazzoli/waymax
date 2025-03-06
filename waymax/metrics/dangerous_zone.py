import chex
import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.utils import geometry


class DangerousZoneMetric(abstract_metric.AbstractMetric):
    """Dangerous zone metric.

    This metric returns 1.0 if an object's bounding box is overlapping with
    that of any dangerous zone.
    """

    @jax.named_scope('DangerousZoneMetric.compute')
    def compute(
        self, simulator_state: datatypes.SimulatorState
    ) -> abstract_metric.MetricResult:
        assert simulator_state.dangerous_zone is not None

        current_object_state = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )
        current_zone_state = datatypes.dynamic_slice(
            simulator_state.dangerous_zone,
            simulator_state.timestep,
            1,
            -1,
        )
        
        return self.danger_check(current_object_state, current_zone_state)

    def danger_check(
        self, agent_traj: datatypes.Trajectory, dangerous_zone_traj: datatypes.object_state.DangerousZone
    ) -> abstract_metric.MetricResult:
        """
        Args:
        agent_traj: Trajectory object containing current states of shape (...,
            num_objects, num_timesteps=1).
        zone_traj: DangerousZone object containing current states of shape (...,
            num_zones, num_timesteps=1).

        Returns:
        A (..., num_objects) MetricResult.
        """
        # (..., n_agent, 6)
        agent_traj_6dof = agent_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw', 'valid'])[..., 0, :]
        # (..., n_zone, 6)
        dangerous_zone_traj_6dof = dangerous_zone_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw', 'valid'])[..., 0, :]

        def unbatched_danger_check(agent_traj: jax.Array, zone_traj: jax.Array) -> jax.Array:
            chex.assert_rank(agent_traj, 2)
            chex.assert_rank(zone_traj, 2)

            danger_check_fn = jax.vmap(geometry.has_overlap, (-2, None), -1)
            danger_check_fn = jax.vmap(danger_check_fn, (None, -2), -1)
            # (n_agent, n_zone)
            danger_condition = danger_check_fn(agent_traj[:,:-1], zone_traj[:,:-1])

            danger_condition = jnp.where(
                agent_traj[:, -1, None], danger_condition, False
            )
            danger_condition = jnp.where(
                zone_traj[None, :, -1], danger_condition, False
            )

            return danger_condition
        batched_danger_check = unbatched_danger_check
        for _ in range(len(agent_traj.shape) - 2):
            batched_danger_check = jax.vmap(batched_danger_check)
        
        danger_condition = batched_danger_check(agent_traj_6dof, dangerous_zone_traj_6dof)
        danger_condition = jnp.any(danger_condition, axis=-1).astype(jnp.float32)

        return abstract_metric.MetricResult.create_and_validate(
            danger_condition, agent_traj.valid[..., 0]
        )
