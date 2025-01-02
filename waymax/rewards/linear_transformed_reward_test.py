from collections import defaultdict

import jax.numpy as jnp
import tensorflow as tf

from waymax import config as _config
from waymax.rewards.linear_transformed_reward import LinearTransformedReward
from waymax.utils import test_utils


class LinearTransformedRewardTest(tf.test.TestCase):

    def test_transform_offroad(self):
        reward_config = _config.LinearTransformedRewardConfig(
                rewards={
                    "offroad": 0.1,
                },
                transform=defaultdict(
                        lambda: lambda x: x,
                        offroad=lambda x: jnp.minimum(x, 0.5),
                ),
        )

        reward = LinearTransformedReward(reward_config)

        # Set up mock simulation state and agent mask
        simulator_state = test_utils.simulator_state_with_offroad()
        agent_mask = jnp.array([1, 1, 1])  # Assume all agents are active

        # Compute the reward
        result = reward.compute(simulator_state, None, agent_mask)

        # Simulating reward metric masked_values as 1.0 for simple example
        offroad_metric = jnp.array([1.0])  # Sample masked values

        # Apply transform and rewards calculation
        capped_values = jnp.minimum(offroad_metric, 0.5)
        expected_reward = capped_values * 0.1  # Reward weight for "offroad_metric"
        self.assertTrue(jnp.allclose(result, expected_reward))



# Run the tests
if __name__ == "__main__":
    tf.test.main()