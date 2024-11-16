import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult
from waymax.metrics.roadgraph import OffroadMetric
from waymax.utils.geometry import wrap_yaws


class GokartProgressMetric(abstract_metric.AbstractMetric):

    @jax.named_scope('GokartProgressMetric.compute')
    def compute(self, state: datatypes.GoKartSimState) -> MetricResult:
        """
        Computes the progress happened in the last step of the trajectory [timestamp-1, timestamp].
        """

        centerline = state.sdc_paths
        if centerline is None:
            raise ValueError(
                    'SimulatorState.sdc_paths required to compute the route progression '
                    'metric.'
            )

        # Shape: (..., num_objects, num_timesteps=1, 2)
        obj_xy_last = datatypes.dynamic_slice(
                state.sim_trajectory.xy,
                start_index=state.timestep - 1,
                slice_size=1,
                axis=-2,
        )
        obj_xy_curr = datatypes.dynamic_slice(
                state.sim_trajectory.xy,
                start_index=state.timestep,
                slice_size=1,
                axis=-2,
        )

        # Shape: (..., 2)
        sdc_xy_last = datatypes.select_by_onehot(
                obj_xy_last[..., 0, :],
                state.object_metadata.is_sdc,
                keepdims=False,
        )
        sdc_xy_curr = datatypes.select_by_onehot(
                obj_xy_curr[..., 0, :],
                state.object_metadata.is_sdc,
                keepdims=False,
        )

        # Shape: (..., num_paths, num_points_per_path)
        dist2centerline = jnp.linalg.norm(
                centerline.xy - jnp.expand_dims(sdc_xy_curr, axis=(-2, -3)),
                axis=-1,
                keepdims=False,
        )
        # # Only consider valid on-route paths.
        # dist = jnp.where(sdc_paths.valid & sdc_paths.on_route, dist_raw, jnp.inf)
        # # Only consider valid SDC states.
        # dist = jnp.where(
        #     jnp.expand_dims(sdc_valid_curr, axis=(-1, -2)), dist, jnp.inf
        # )

        # (..., num_paths, 1) find the nearest point to the car on each path
        dist_path = jnp.min(dist2centerline, axis=-1, keepdims=True)
        # (..., 1, 1) find the index of the nearest path
        idx = jnp.argmin(dist_path, axis=-2, keepdims=True)
        # (...) find the minimum distance to the nearest path
        min_dist_path = jnp.min(dist2centerline, axis=(-1, -2))

        # Shape: (..., max(num_points_per_path))
        ref_path = jax.tree_util.tree_map(
                lambda x: jnp.take_along_axis(x, indices=idx, axis=-2)[..., 0, :],
                centerline,
        )

        def get_arclength_for_pts(xy: jax.Array, path: datatypes.Paths):
            # Shape: (..., max(num_points_per_path))
            dist_raw = jnp.linalg.norm(
                    xy[..., jnp.newaxis, :] - path.xy, axis=-1, keepdims=False
            )
            dist = jnp.where(path.valid, dist_raw, jnp.inf)
            idx = jnp.argmin(dist, axis=-1, keepdims=True)
            # (..., )
            return jnp.take_along_axis(path.arc_length, indices=idx, axis=-1)[..., 0], idx

        last_dist, last_idx = get_arclength_for_pts(sdc_xy_last, ref_path)
        curr_dist, curr_idx = get_arclength_for_pts(sdc_xy_curr, ref_path)

        # (..., num_paths=1, 1, 2) find the direction of the centerline at the nearest point
        dir_ref = jnp.take_along_axis(state.sdc_paths.dir_xy.squeeze(-3), curr_idx[..., None], axis=-2)
        dir_ref = jnp.squeeze(dir_ref, axis=-2)  # (...,2)
        
        # Normalized one by waymo
        # progress = jnp.where(
        #     end_dist == start_dist,
        #     FULL_PROGRESS_VALUE,
        #     (curr_dist - start_dist) / (end_dist - start_dist),
        # )
        # Progress in [m]
        progress = curr_dist - last_dist
        valid = jnp.isfinite(min_dist_path)
        progress = jnp.where(valid, progress, 0.0)
        # movement vector between the last and current position of sdc
        movement_vector = sdc_xy_curr - sdc_xy_last
        movement_vector /= jnp.linalg.norm(movement_vector)
        # Decreased reward if the movement is not "aligned" with the track tangent
        # In particular to avoid crossing the finish line backwards and getting a high reward
        alignment = jnp.dot(movement_vector, dir_ref)
        progress = jnp.where(
                alignment > 0.7,  # ~= cos45 around 45 degree
                progress,
                0)
        path_length = state.sdc_paths.arc_length[..., 0, -1]

        # check if the car has reached the end of the path (i.e., it has completed a lap)
        # (in this case, the progress is negative, so we need to add the path length)
        # a small progress 0.1 between the last point and the first point of the path
        progress = jnp.where(progress < -path_length / 2, path_length + progress + 0.1, progress)

        progress = jnp.where(state.timestep <= 0, jnp.zeros(state.sim_trajectory.x.shape[:-2]), progress)
        # print(f"progress: {progress}")
        return MetricResult.create_and_validate(
                value=progress,
                valid=jnp.ones(progress.shape, dtype=bool))


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
        orientation_reward = jnp.exp(-dir_diff ** 2 / 0.5)
        # scaled by the velocity, negative if the car is moving in the opposite direction
        orientation_reward *= jnp.tanh(sdc_vel_curr[0])  # (...,) vx
        #az: maybe tanh instead of clipping?
        orientation_reward = jnp.clip(orientation_reward, -1, 1) # 0.05

        return MetricResult.create_and_validate(
                value=orientation_reward,
                valid=jnp.ones(orientation_reward.shape, dtype=jnp.bool)
        )


class GokartOffroadMetric(OffroadMetric):

    @jax.named_scope('GokartOffroadMetric.compute')
    def compute(self, state: datatypes.GoKartSimState) -> MetricResult:
        """Same as the OffroadMetric but with float32 dtype."""
        is_offroad = super().compute(state)
        # fixme remove player dimension (to be coherent with all the other gokart metrics,
        #  but not ideal for multiagent envs)
        return is_offroad.replace(
                value=jnp.squeeze(is_offroad.value, axis=-1),
                valid=jnp.squeeze(is_offroad.valid, axis=-1)
        )
