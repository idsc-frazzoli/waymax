# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic parameters based datastructures for Waymax."""
from typing import Sequence

import chex
import jax
from jax import numpy as jnp

@chex.dataclass
class DynamicsParams:
    pass

@chex.dataclass
class GokartDynamicsParams(DynamicsParams):
        
  Iz: jax.Array
  front_paj_B: jax.Array
  front_paj_C: jax.Array
  front_paj_D: jax.Array
  front_paj_E: jax.Array
  rear_paj_B: jax.Array
  rear_paj_C: jax.Array
  rear_paj_D: jax.Array
  rear_paj_E: jax.Array
        
  @classmethod
  def zeros(cls, shape: Sequence[int]) -> "GokartDynamicsParams":
        """Creates a GokartDynamicsParams containing default values."""
        return cls(
            Iz=jnp.zeros(shape, jnp.float32),
            front_paj_B=jnp.zeros(shape, jnp.float32),
            front_paj_C=jnp.zeros(shape, jnp.float32),
            front_paj_D=jnp.zeros(shape, jnp.float32),
            front_paj_E=jnp.zeros(shape, jnp.float32),
            rear_paj_B=jnp.zeros(shape, jnp.float32),
            rear_paj_C=jnp.zeros(shape, jnp.float32),
            rear_paj_D=jnp.zeros(shape, jnp.float32),
            rear_paj_E=jnp.zeros(shape, jnp.float32),
        )