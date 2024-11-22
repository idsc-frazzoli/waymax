import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult
from waymax.utils.geometry import wrap_yaws


class GokartOrientationMetric(abstract_metric.AbstractMetric):

    @jax.named_scope('GokartOrientationMetric.compute')
    def compute(self, state: datatypes.GoKartSimState) -> MetricResult:
        """
        Computes the orientation reward. The car is rewarded for moving in the direction of the nearest point on the reference track(centerline).

        Args:
        state: The current state of the simulator.

        Returns:
        The orientation reward.
        """

        centerline = state.sdc_paths
        if centerline is None:
            raise ValueError(
                    'SimulatorState.sdc_paths required to compute the orientation reward '
                    'metric.'
            )
        # Shape: (..., num_objects, num_timesteps=1, 2)
        obj_xy_curr = datatypes.dynamic_slice(
                state.sim_trajectory.xy,
                start_index=state.timestep,
                slice_size=1,
                axis=-2,
        )

        # Shape: (..., 2)
        sdc_xy_curr = datatypes.select_by_onehot(
                obj_xy_curr[..., 0, :],
                state.object_metadata.is_sdc,
                keepdims=False,
        )

        # Shape: (..., num_paths=1, num_points_per_path)
        dist2centerline = jnp.linalg.norm(
                centerline.xy - jnp.expand_dims(sdc_xy_curr, axis=(-2, -3)),
                axis=-1,
                keepdims=False,
        )

        # (..., num_paths=1, 1) find the index of the nearest point on the centerline
        idx = jnp.argmin(dist2centerline, axis=-1, keepdims=True)

        # (..., num_paths=1, 1, 2) find the direction of the centerline at the nearest point
        dir_ref = jnp.take_along_axis(state.sdc_paths.dir_xy, idx[..., None], axis=-2)
        dir_ref = jnp.squeeze(dir_ref, axis=(-2, -3))  # (...,2)

        yaw_ref = wrap_yaws(jnp.arctan2(dir_ref[..., 1], dir_ref[..., 0]))  # (...,)

        # shape: (..., num_objects, timesteps=1, 2) -> (..., num_objects, 2)
        vel_xy = state.current_sim_trajectory.vel_xy[..., 0, :]

        # shape: (...,2)
        sdc_vel_curr = datatypes.select_by_onehot(
                vel_xy,
                state.object_metadata.is_sdc,
                keepdims=False,
        )

        # shape: (..., num_objects, timesteps=1) -> (..., num_objects)
        yaw = state.current_sim_trajectory.yaw[..., 0]

        sdc_yaw_curr = datatypes.select_by_onehot(
                yaw,
                state.object_metadata.is_sdc,
                keepdims=False,
        )
        # yaw_vector = jnp.array([jnp.cos(sdc_yaw_curr), jnp.sin(sdc_yaw_curr)])  # (..., 2)
        dir_diff = jnp.abs(wrap_yaws(yaw_ref - sdc_yaw_curr))  # (...,)
        # encourage the car to move in the direction of the reference track(centerline)
        # orientation_reward = jnp.dot(yaw_vector, dir_ref)  # (...,)
        # orientation_reward = jnp.where(orientation_reward > 0, orientation_reward, 0)
        orientation_reward = jnp.exp(-dir_diff ** 2 / 0.5)
        # scaled by the velocity, negative if the car is moving in the opposite direction
        orientation_reward *= jnp.tanh(sdc_vel_curr[0])  # (...,) vx
        #az: maybe tanh instead of clipping?
        orientation_reward = jnp.clip(orientation_reward, -1, 1) # 0.05

        return MetricResult.create_and_validate(
                value=orientation_reward,
                valid=jnp.ones(orientation_reward.shape, dtype=jnp.bool)
        )



