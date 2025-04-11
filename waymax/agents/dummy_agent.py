from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from waymax import datatypes
from waymax.agents import actor_core
from waymax.agents.expert import _IS_SDC_FUNC

_DUMMY_NAME = "dummy"


def create_dummy_agent(
    actions: Float[Array, "T d"],
    initial_timestep: Int[Array, ""] = jnp.array(0, dtype=jnp.int32),
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array] = _IS_SDC_FUNC,
) -> actor_core.WaymaxActorCore:
    """
    Creates a dummy agent using the WaymaxActorCore interface.

    A dummy agent takes predefined actions, and parses it into a WaymaxActorOutput.

    Assumes that the state.timestep starts at 0, and increments by 1 for each step.

    Args:
        dynamics_model: The dynamics model to use.
        is_controlled_func: The function that determines whether an object is controlled.

    Returns:
        A dummy agent.
    """

    def select_action(  # pytype: disable=annotation-type-mismatch
        params: actor_core.Params,
        state: datatypes.SimulatorState,
        actor_state: actor_core.ActorState,
        rng: jax.Array | None = None,
    ) -> actor_core.WaymaxActorOutput:
        """Provided with an action, returns a WaymaxActorOutput with the action and is_controlled set to True."""
        del params, rng, actor_state  # unused.
        valid = is_controlled_func(state)
        valid = jnp.reshape(valid, (-1, 1))

        # get action of single agent
        action = actions[state.timestep - initial_timestep, :]

        # tile actions to all agents since the other agents are not controlled. Shape (num_objects, action_dim)
        action = jnp.tile(action, (state.num_objects, 1))

        # ensure that only the controlled agents have actions non-zero actions.
        action = datatypes.Action(data=action * valid, valid=valid)
        return actor_core.WaymaxActorOutput(
            actor_state=None,  # type: ignore
            action=action,  # type: ignore
            is_controlled=is_controlled_func(state),  # type: ignore
        )

    return actor_core.actor_core_factory(
        init=lambda rng, init_state: None,
        select_action=select_action,
        name=_DUMMY_NAME,
    )
