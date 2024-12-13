import tensorflow as tf
from absl.testing import parameterized

from gocarx.env.track_config import TrackConfig, TrackType
from gocarx.utils.gokart_utils import create_init_state
from waymax.metrics import GokartOffroadMetric, GokartDistanceToBoundsMetric


class GokartOffroadMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_onroad(self):
        metric = GokartOffroadMetric()
        state = create_init_state(num_timesteps=100)
        result = metric.compute(state)
        # should be zero, because the car is not offroad
        self.assertEqual(result.value, 0.0)

    def test_offroad(self):
        metric = GokartOffroadMetric()
        state = create_init_state(num_timesteps=100)
        current_y = state.current_sim_trajectory.x[..., 0, 0]
        # move the car offroad
        current_y -= 2
        state.sim_trajectory.y = state.sim_trajectory.y.at[..., 0, 0].set(current_y)
        result = metric.compute(state)
        # should be negative, because the car is offroad
        self.assertEqual(result.value, 1.0)
        
class GokartDistanceToBoundsMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test(self):
        state = create_init_state(num_timesteps=5, track_config=TrackConfig(TrackType.WINTI_TEST_AIDED_3, False))
        
        metric = GokartDistanceToBoundsMetric()
        result1 = metric.compute(state)
        self.assertAllGreater(result1.value, 0.0)
        
        state.sim_trajectory.y += -1.25
        metric = GokartDistanceToBoundsMetric(offroad_value=-.5)
        result = metric.compute(state)
        self.assertAllGreater(result.value,result1.value)
        
        state.sim_trajectory.y += -0.5
        metric = GokartDistanceToBoundsMetric(offroad_value=-.5)
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.504138)

        
        state.sim_trajectory.y += -5.0
        metric = GokartDistanceToBoundsMetric(offroad_value=-1)
        result = metric.compute(state)
        self.assertAllClose(result.value, -5.25)
        
        metric = GokartDistanceToBoundsMetric()
        result = metric.compute(state)
        self.assertAllLess(result.value, 0)
        

if __name__ == "__main__":
    tf.test.main()
        
        
        