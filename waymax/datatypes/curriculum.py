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
class CurriculumParams:
    pass

@chex.dataclass
class GokartCurriculumParams(CurriculumParams):
            
    level: jax.Array
        
    def validate(self):
        """Validates shape and type."""
        chex.assert_equal_shape(
            [
                self.level,
            ]
        )
        chex.assert_type(
            [
                self.level,
            ],
            [
                jnp.integer,
            ],
        )
                