import tensorflow as tf
from absl.testing import parameterized

from gocarx.utils.gokart_utils import create_init_state
from waymax.metrics import GokartStateNormMetric, GokartStateOutRangeMetric, GokartVelyMetric, GokartVelxOutRangeMetric


class GokartStateMetricTest(tf.test.TestCase, parameterized.TestCase):
    
    def test(self):
        state = create_init_state(num_timesteps=5)
        state.sim_trajectory.yaw_rate = state.sim_trajectory.yaw_rate.at[:].set(-1.234)
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[:].set(6.937)
        
        metric = GokartStateNormMetric("yaw_rate")
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.522756)
        
        metric = GokartStateNormMetric("vel_x")
        result = metric.compute(state)
        self.assertAllClose(result.value, 48.121967)
        
        
class GokartVelyMetricTest(tf.test.TestCase, parameterized.TestCase):
        
        def test(self):
            state = create_init_state(num_timesteps=5)
            state.sim_trajectory.vel_y = state.sim_trajectory.vel_y.at[:].set(6.937)
            
            metric = GokartVelyMetric()
            result = metric.compute(state)
            self.assertAllClose(result.value, 48.121967)
            
            state.sim_trajectory.vel_y = state.sim_trajectory.vel_y.at[:].set(-2.593)
            
            metric = GokartVelyMetric()
            result = metric.compute(state)
            self.assertAllClose(result.value, 6.723649)
        
        
class GokartStateOutRangeMetricTest(tf.test.TestCase, parameterized.TestCase):
    
    def test(self):
        state = create_init_state(num_timesteps=5)
        state.sim_trajectory.yaw_rate = state.sim_trajectory.yaw_rate.at[:].set(-1.234)
        state.sim_trajectory.vel_y = state.sim_trajectory.vel_y.at[:].set(6.937)
        
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
        
        
class GokartVelxOutRangeMetricTest(tf.test.TestCase, parameterized.TestCase):
    
    def test(self):
        state = create_init_state(num_timesteps=5)
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[:].set(6.937)
        
        metric = GokartStateOutRangeMetric("vel_x", max_value=6.4)
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)
        
        metric = GokartStateOutRangeMetric("vel_x", max_value=7)
        result = metric.compute(state)
        self.assertEqual(result.value, 1.0)
        

        
        
if __name__ == "__main__":
    tf.test.main()