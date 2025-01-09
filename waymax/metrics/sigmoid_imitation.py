import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric


class SigmoidLogDivergenceMetric(abstract_metric.AbstractMetric):
    """Log divergence metric transformed by sigmoid.

    This metric transforms the L2 distance between the controlled object's XY
    location and its position in the logged history at the same timestep into
    a limited range by using an adapted sigmoid function.
    """

    def __init__(self, translation: float = 2.5):
        """
        Args:
            translation: the flipped sigmoid can be translated in the positive
                direction of x-axis, and the variable indicates the amount of
                translation.
        """
        assert isinstance(translation, (float, int))
        assert translation>0
        self._translation = translation

    @jax.named_scope('SigmoidLogDivergenceMetric.compute')
    def compute(
        self, simulator_state: datatypes.SimulatorState
    ) -> abstract_metric.MetricResult:
        """
        Same as the LogDivergenceMetric, but copied here for potential
        modification to include yaw difference.
        """
        current_object_state = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )
        current_log_state = datatypes.dynamic_slice(
            simulator_state.log_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )
        result = self.compute_sigmoid_log_divergence(
            current_object_state.xy, current_log_state.xy, self._translation
        )
        valid = current_object_state.valid & current_log_state.valid
        return abstract_metric.MetricResult.create_and_validate(
            result[..., 0], valid[..., 0]
        )

    @classmethod
    def compute_sigmoid_log_divergence(
        cls, object_xy: jax.Array, log_xy: jax.Array, translation: float
    ) -> jax.Array:

        dist = jnp.linalg.norm(object_xy - log_xy, axis=-1)
        transformed_dist = jax.nn.sigmoid(-(dist-translation))
        return transformed_dist
