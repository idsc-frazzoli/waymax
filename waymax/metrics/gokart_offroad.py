import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult
from waymax.metrics.roadgraph import OffroadMetric
from waymax.utils.geometry import wrap_yaws


class GokartOffroadMetric(OffroadMetric):

    @jax.named_scope('GokartOffroadMetric.compute')
    def compute(self, state: datatypes.GoKartSimState) -> MetricResult:
        """Same as the OffroadMetric but with float32 dtype."""
        is_offroad = super().compute(state)
        # fixme remove player dimension (to be coherent with all the other gokart metrics,
        #  but not ideal for multiagent envs)
        return is_offroad.replace(
                value=jnp.squeeze(is_offroad.value, axis=-1),
                valid=jnp.squeeze(is_offroad.valid, axis=-1)
        )