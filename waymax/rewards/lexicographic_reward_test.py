import jax.numpy as jnp
import tensorflow as tf

from waymax import config as _config
from waymax.rewards.lexicographic_reward import LexicographicReward
from waymax.utils import test_utils


class LexicographicRewardTest(tf.test.TestCase):

    def test_lexicographic_reward(self):
        reward_config = _config.LexicographicRewardConfig(
            rewards={
                "offroad": -10.0,
                "log_divergence": 1.0,
            },
            hierarchy={
                "offroad": 1,
                "log_divergence": 2,
            },
            num_hierarchies=2,
        )

        reward = LexicographicReward(reward_config)

        # Set up mock simulation state and agent mask
        simulator_state = test_utils.simulator_state_with_offroad()
        agent_mask = jnp.array([1, 1, 1])  # Assume all agents are active

        # Compute the reward
        result = reward.compute(simulator_state, None, agent_mask)

        # Format the reward
        expected_reward = jnp.array([[-10.0, 0.0],[-10.0, 0.0],[-10.0, 0.0]])
        self.assertTrue(jnp.allclose(result, expected_reward))


# Run the tests
if __name__ == "__main__":
    tf.test.main()