import jax.numpy as jnp
import tensorflow as tf
from waymax import dataloader
from waymax import datatypes
from waymax.metrics import dangerous_zone
from waymax.utils import test_utils

from absl.testing import parameterized


class DangerousZoneMetricTest(tf.test.TestCase, parameterized.TestCase):

  def test_metric_runs_from_real_data(self):
    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    sim_state_init = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )
    dangerous_zone = datatypes.object_state.DangerousZone(
        x = 0.5*jnp.ones((1,91), jnp.float32),
        y = 1.0*jnp.ones((1,91), jnp.float32),
        yaw = 0*jnp.ones((1,91), jnp.float32),
        valid = jnp.ones((1,91), jnp.bool_),
        length = 2.0*jnp.ones((1,91), jnp.float32),
        width = 1.0*jnp.ones((1,91), jnp.float32),
    )
    sim_state_init = sim_state_init.replace(dangerous_zone = dangerous_zone)
    result = dangerous_zone.DangerousZoneMetric().compute(sim_state_init)
    self.assertEqual(result.value.shape, (128,))
    self.assertEqual(result.valid.shape, (128,))

  def test_metric_detects_two_agents_in_danger(self):
    traj_with_no_overlaps = test_utils.simulated_trajectory_no_overlap()
    dangerous_zone_slice = datatypes.object_state.DangerousZone(
        x = 0.5*jnp.ones((1,1), jnp.float32),
        y = 1.0*jnp.ones((1,1), jnp.float32),
        yaw = 0*jnp.ones((1,1), jnp.float32),
        valid = jnp.ones((1,1), jnp.bool_),
        length = 2.0*jnp.ones((1,1), jnp.float32),
        width = 1.0*jnp.ones((1,1), jnp.float32),
    )
    metric = dangerous_zone.DangerousZoneMetric().compute_overlap(traj_with_no_overlaps, dangerous_zone_slice)
    num_objects = traj_with_no_overlaps.num_objects
    with self.subTest('value'):
      self.assertAllEqual(
          metric.value,
          jnp.array(
              [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          ),
      )
    with self.subTest('valid'):
      self.assertAllEqual(metric.valid, traj_with_no_overlaps.valid[..., 0])

  def test_metric_detects_no_agent_in_danger(self):
    traj_with_no_overlaps = test_utils.simulated_trajectory_no_overlap()
    dangerous_zone_slice = datatypes.object_state.DangerousZone(
        x = 0.5*jnp.ones((1,1), jnp.float32),
        y = 5.0*jnp.ones((1,1), jnp.float32),
        yaw = 0*jnp.ones((1,1), jnp.float32),
        valid = jnp.ones((1,1), jnp.bool_),
        length = 2.0*jnp.ones((1,1), jnp.float32),
        width = 1.0*jnp.ones((1,1), jnp.float32),
    )
    metric = dangerous_zone.DangerousZoneMetric().compute_overlap(traj_with_no_overlaps, dangerous_zone_slice)
    num_objects = traj_with_no_overlaps.num_objects
    with self.subTest('value'):
      self.assertAllEqual(
          metric.value,
          jnp.zeros(
              num_objects,
          ),
      )
    with self.subTest('valid'):
      self.assertAllEqual(metric.valid, traj_with_no_overlaps.valid[..., 0])


if __name__ == '__main__':
  tf.test.main()