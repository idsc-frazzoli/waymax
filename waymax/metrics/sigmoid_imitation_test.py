import jax
from jax import numpy as jnp
import tensorflow as tf
from absl.testing import parameterized

from waymax import dataloader
from waymax import datatypes
from waymax.metrics import sigmoid_imitation
from waymax.utils import test_utils


class SigmoidLogDivergenceMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_metric_runs_from_real_data(self):
        dataset = test_utils.make_test_dataset()
        data_dict = next(dataset.as_numpy_iterator())
        sim_state_init = dataloader.simulator_state_from_womd_dict(
            data_dict, time_key='all'
        )
        result = sigmoid_imitation.SigmoidLogDivergenceMetric().compute(sim_state_init)
        self.assertEqual(result.value.shape, (128,))
        self.assertEqual(result.valid.shape, (128,))

    @parameterized.parameters(((1,),), ((3, 5),), ((6, 8, 9),))
    def test_metric_returns_correct_results(self, dimensions):
        object_state = datatypes.Trajectory.zeros(dimensions).replace(
            x=jnp.ones(dimensions),
            y=jnp.ones(dimensions),
        )
        log_state = object_state.replace(
            x=jnp.ones(dimensions) * 3.0, y=jnp.ones(dimensions) * 1.0
        )
        temp_class = sigmoid_imitation.SigmoidLogDivergenceMetric()
        result = temp_class.compute_sigmoid_log_divergence(
            object_state.xy, log_state.xy, temp_class._translation
        )
        expected = jax.nn.sigmoid(-(jnp.ones(dimensions)*2.0 - temp_class._translation))
        self.assertAllClose(result, expected)


if __name__ == '__main__':
  tf.test.main()