import jax.numpy as jnp

from waymax.datatypes import Action, GokartTrajectory
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.utils.gokart_config import GoKartGeometry, TricycleParams, PajieckaParams


def test_dynamics() :
    # todo implement real tests
  gk_geometry = GoKartGeometry()
  model_params = TricycleParams()
  paj_params = PajieckaParams()
  tricycle = TricycleModel(gk_geometry, model_params, paj_params)
  action = Action(data=jnp.array([[0.1, 0.1, 0.1],]), valid=jnp.array([True, True, True]))
  trajectory = GokartTrajectory.zeros((1,1,))
  print(tricycle.compute_update(action, trajectory))


def test_dynamics_with_batch_dimension():
    pass