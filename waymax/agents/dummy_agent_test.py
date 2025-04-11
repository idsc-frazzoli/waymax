import jax.numpy as jnp
import pytest

from waymax.agents.dummy_agent import create_dummy_agent
from waymax import dataloader
from waymax.utils import test_utils


def test_dummy_agent():
    # initialize the scenario
    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    scenario = dataloader.simulator_state_from_womd_dict(data_dict, time_key="all")  # type: ignore

    # sdc
    i = jnp.where(scenario.object_metadata.is_sdc)[0][0]

    # get action
    actions = jnp.array([[1.0, 0.0], [1.0, 1.0]])
    dummy_actor = create_dummy_agent(actions, is_controlled_func=lambda state: scenario.object_metadata.is_sdc)

    action = dummy_actor.select_action({}, scenario, jnp.zeros(2), jnp.zeros(2))
    assert jnp.all(action.action.data[i, :] == actions[0, :]), f"Expected action: {actions[0, :]}, got: {action}"

    # set timestep = 1
    scenario.timestep = 1
    action = dummy_actor.select_action({}, scenario, jnp.zeros(2), jnp.zeros(2))

    assert jnp.all(
        action.action.data[i, :] == actions[1, :]
    ), f"Expected action: {action.action.data[i, :]}, got: {action}"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
    # test_dummy_agent()
