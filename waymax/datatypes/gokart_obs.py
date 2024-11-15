import chex
import jax
import jax.numpy as jnp


@chex.dataclass
class GokartObservation:
  """Gokart Observation at a single simulation step. (inspired by Observation Class from Waymax)

  The observation can include a fixed number of history information. Note 
  this is only considering the single agent case.

  Attributes:
    vel_x: Longitudinal velocity in body frame of shape (..., 1).
    vel_y: Lateral velocity in body frame of shape (..., 1).
    vel_r: Angular velocity in body frame of shape (..., 1).
    dir_diff: Difference between orientation of object and reference 
      direction of shape (..., 1).
    dist_to_edge: Distance to the boundary of the track of shape (..., 10).
  """

  vel_x:jax.Array
  vel_y:jax.Array
  vel_r:jax.Array
  dir_diff:jax.Array
  dist_to_edge:jax.Array

  def flatten(self):
    """Return a flattened jax Array of the observation attributes."""
    return jnp.concatenate(
        [self.vel_x, self.vel_y, self.vel_r, self.dir_diff, self.dist_to_edge], axis=-1
    )