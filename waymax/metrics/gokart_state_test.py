import tensorflow as tf
from absl.testing import parameterized

from gocarx.utils.gokart_utils import create_init_state
from waymax.metrics import GokartStateMetric, GokartStateOutRangeMetric


class GokartStateMetricTest(tf.test.TestCase, parameterized.TestCase):
    
    def test(self):
        state = create_init_state(num_timesteps=5)
        state.sim_trajectory.yaw_rate = state.sim_trajectory.yaw_rate.at[:,2].set(-1.234)
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[:,2].set(6.937)
        state.sim_trajectory.vel_y = state.sim_trajectory.vel_y.at[:,2].set(-2.593)
        
        state.timestep = 2
        
        metric = GokartStateMetric("yaw_rate")
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.522756)
        
        metric = GokartStateMetric("vel_x")
        result = metric.compute(state)
        self.assertAllClose(result.value, 48.121967)
        
        metric = GokartStateMetric("vel_y")
        result = metric.compute(state)
        self.assertAllClose(result.value, 6.723649)
        
class GokartStateOutRangeMetricTest(tf.test.TestCase, parameterized.TestCase):
    
    def test(self):
        state = create_init_state(num_timesteps=5)
        state.sim_trajectory.yaw_rate = state.sim_trajectory.yaw_rate.at[:,2].set(-1.234)
        state.sim_trajectory.vel_y = state.sim_trajectory.vel_y.at[:,2].set(6.937)
        
        state.timestep = 2
        
        metric = GokartStateOutRangeMetric("yaw_rate")
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)
        
        metric = GokartStateOutRangeMetric("yaw_rate", 1.0)
        result = metric.compute(state)
        self.assertEqual(result.value, 1.0)
        
        metric = GokartStateOutRangeMetric("yaw_rate", -1.5)
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)
        
        metric = GokartStateOutRangeMetric("vel_y", max_value=6.0)
        result = metric.compute(state)
        self.assertEqual(result.value, 1.0)
        
        metric = GokartStateOutRangeMetric("vel_y", max_value=7.5)
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)
        
        
if __name__ == "__main__":
    tf.test.main()