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
    m: jax.Array
    h: jax.Array
    front_paj_B: jax.Array
    front_paj_C: jax.Array
    front_paj_D: jax.Array
    front_paj_E: jax.Array
    rear_paj_B: jax.Array
    rear_paj_C: jax.Array
    rear_paj_D: jax.Array
    rear_paj_E: jax.Array
        
    @classmethod
    def create_from_configs(
        cls, shape: Sequence[int],
        gk_geometry: 'GokartGeometry', model_params: 'GokartParams', paj_params: 'PajieckaParams'
    ) -> "GokartDynamicsParams":
        """Creates a GokartDynamicsParams containing the given values."""
        Iz = model_params.Iz
        m = gk_geometry.m
        h = gk_geometry.h
        front_paj_B = paj_params.front_paj.B
        front_paj_C = paj_params.front_paj.C
        front_paj_D = paj_params.front_paj.D
        front_paj_E = paj_params.front_paj.E
        rear_paj_B = paj_params.rear_paj.B
        rear_paj_C = paj_params.rear_paj.C
        rear_paj_D = paj_params.rear_paj.D
        rear_paj_E = paj_params.rear_paj.E
        return cls(
            Iz=Iz*jnp.ones(shape, jnp.float32),
            m=m*jnp.ones(shape, jnp.float32),
            h=h*jnp.ones(shape, jnp.float32),
            front_paj_B=front_paj_B*jnp.ones(shape, jnp.float32),
            front_paj_C=front_paj_C*jnp.ones(shape, jnp.float32),
            front_paj_D=front_paj_D*jnp.ones(shape, jnp.float32),
            front_paj_E=front_paj_E*jnp.ones(shape, jnp.float32),
            rear_paj_B=rear_paj_B*jnp.ones(shape, jnp.float32),
            rear_paj_C=rear_paj_C*jnp.ones(shape, jnp.float32),
            rear_paj_D=rear_paj_D*jnp.ones(shape, jnp.float32),
            rear_paj_E=rear_paj_E*jnp.ones(shape, jnp.float32),
        )
        
    def validate(self):
        """Validates shape and type."""
        chex.assert_equal_shape(
            [
                self.Iz,
                self.m,
                self.h,
                self.front_paj_B,
                self.front_paj_C,
                self.front_paj_D,
                self.front_paj_E,
                self.rear_paj_B,
                self.rear_paj_C,
                self.rear_paj_D,
                self.rear_paj_E,
            ]
        )
        chex.assert_type(
            [
                self.Iz,
                self.m,
                self.h,
                self.front_paj_B,
                self.front_paj_C,
                self.front_paj_D,
                self.front_paj_E,
                self.rear_paj_B,
                self.rear_paj_C,
                self.rear_paj_D,
                self.rear_paj_E,
            ],
            [
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
            ],
        )
                