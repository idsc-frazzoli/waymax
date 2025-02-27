import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric


class PenalizeHeadbackMetric(abstract_metric.AbstractMetric):
    """Penalize headback metric.

    This metric returns 1.0 if the velocity along the local x-axis
    takes negative value.
    """

    @jax.named_scope('PenalizeHeadbackMetric.compute')
    def compute(
        self, simulator_state: datatypes.SimulatorState
    ) -> abstract_metric.MetricResult:
        
        current_object_state = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )

        current_angle_diff = current_object_state.yaw-jnp.pi/2
        current_local_vel_x = jnp.cos(current_angle_diff)*current_object_state.vel_y - jnp.sin(current_angle_diff)*current_object_state.vel_x

        result = (current_local_vel_x<0).astype(jnp.float32)
        valid = jnp.ones_like(result, dtype=jnp.bool_)

        return abstract_metric.MetricResult.create_and_validate(
            result[..., 0], valid[..., 0]
        )
