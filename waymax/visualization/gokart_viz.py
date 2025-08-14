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

"""Visualization functions for Waymax data structures."""

from functools import partial
from typing import Any, Iterable, List, Optional
import typing

import jax
import matplotlib

from matplotlib.animation import FuncAnimation
import numpy as np

from waymax import config as waymax_config
from waymax import datatypes
from waymax.datatypes import operations
from waymax.utils import geometry
from waymax.visualization import color
from waymax.visualization import utils
from waymax.visualization import viz

import matplotlib.pyplot as plt

GokartObs = typing.TypeVar("GokartObs")


def create_video_simulator_state(
    state: datatypes.SimulatorState,
    obs: GokartObs | None = None,
    video_path: str | None = None,
    use_log_traj: bool = True,
    n_steps: int = 10,
    interval: int = 100,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    ref: bool = False,
    viz_config: Optional[dict[str, Any]] = None,
) -> List[np.ndarray]:
    """
    Make an animation of the simulator state. Return the list of numpy matrices representing the video frames if
    video_path is None. Otherwise, the video is saved to disk at video_path path and an empty list is returned.

    Retuning the list of numpy matrices is faster (2-3x) than saving the video to disk.
    """

    video_plotter = VideoPlotter(
        video_path, viz_config, skip_traffic_light=True, plot_last_history_only=True, faster_axis_origin=True
    )

    imgs = video_plotter.plot_sequence_simulator_state(
        state,
        obs,
        use_log_traj,
        n_steps,
        interval,
        batch_idx,
        highlight_obj,
        ref,
    )

    return imgs


class VideoPlotter:

    def __init__(
        self,
        video_path: str | None = None,
        viz_config: Optional[dict[str, Any]] = None,
        skip_traffic_light: bool = False,
        plot_last_history_only: bool = False,
        faster_axis_origin: bool = False,
    ):
        self.viz_config = utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
        self.video_path = video_path
        self.viz_config_dict = viz_config

        self.skip_traffic_light = skip_traffic_light
        self.plot_last_history_only = plot_last_history_only
        self.faster_axis_origin = faster_axis_origin

        # Initialized like this to remove obnoxious pylance errors when using self.ax.plot
        fig_ax = self.create_fig()
        self.fig: matplotlib.figure.Figure = fig_ax[0]
        self.ax: matplotlib.axes.Axes = fig_ax[1]

        self.trajectory_lines = None
        self.history_lines = None
        self.context_lines = None
        self.overlap_lines = None
        self.reference_lines = None
        self.rays_lines = None
        self.text = None
        self.roadgraph_lines_dict = None

        self.name_trajectory_lines = "trajectory_lines"
        self.name_history_lines = "history_lines"
        self.name_context_lines = "context_lines"
        self.name_overlap_lines = "overlap_lines"

    def __del__(self):
        self.close_fig()

    def create_fig(self) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig, ax = utils.init_fig_ax(self.viz_config)
        # Just enough margin in the figure to display xticks and yticks.
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
        return fig, ax

    def close_fig(self):
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = None, None

    def plot_sequence_simulator_state(
        self,
        state: datatypes.SimulatorState,
        obs: GokartObs | None,
        use_log_traj: bool = True,
        n_steps: int = 10,
        interval: int = 100,
        batch_idx: int = -1,
        highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
        ref: bool = False,
    ) -> List[np.ndarray]:
        """
        Make an animation of the simulator state. Return the list of numpy matrices representing the video frames if
        video_path is None. Otherwise, the video is saved to disk at video_path path and an empty list is returned.

        Args:
          state: A SimulatorState instance.
          obs: Optional observation instance.
          use_log_traj: Set True to use logged trajectory, o/w uses simulated
            trajectory.
          viz_config: dict for optional config.
          batch_idx: optional batch index.
          highlight_obj: Represents the type of objects that will be highlighted with
            `color.COLOR_DICT['controlled']` color.
          ref: Set True to plot reference trajectory.

        Returns:
          list of np images if video_path is None, otherwise an empty list since the video will be saved to disk.
        """
        if self.fig is None or self.ax is None:
            self.create_fig(self.viz_config)
        imgs = []
        if batch_idx > -1:
            if len(state.shape) != 1:
                raise ValueError(f"Expecting one batch dimension, got {len(state.shape)}")
            state = viz._index_pytree(state, batch_idx)
            obs = viz._index_pytree(obs, batch_idx)

        if self.video_path is not None:
            def animate_step(
                i: int,
                state: datatypes.GoKartSimState,
                obs: GokartObs | None,
                use_log_traj: bool,
                highlight_obj: waymax_config.ObjectType,
                ref: bool,
            ) -> Iterable[matplotlib.lines.Line2D]:
                # Print progress every 10% of frames or every 10 frames, whichever is smaller
                progress_interval = max(1, min(10, n_steps // 10))
                if i % progress_interval == 0 or i == n_steps - 1:
                    progress_pct = (i + 1) / n_steps * 100
                    print(f"  📊 Frame {i + 1}/{n_steps} ({progress_pct:.1f}%)")
                
                state = state.replace(timestep=i)
                obs_t = operations.dynamic_slice(obs, i, 1, axis=0) if obs is not None else None

                self.plot_simulator_state(state, obs_t, use_log_traj, highlight_obj, ref)

                artists = []
                for line in [
                    self.trajectory_lines,
                    self.context_lines,
                    self.overlap_lines,
                    self.reference_lines,
                    self.rays_lines,
                ]:
                    if line is not None:
                        artists.extend(line)
                if not self.plot_last_history_only and self.history_lines is not None:
                    artists.extend(self.history_lines)
                if self.text is not None:
                    artists.append(self.text)

                return artists

            partial_animate_step = partial(
                animate_step,
                state=state,
                obs=obs,
                use_log_traj=use_log_traj,
                highlight_obj=highlight_obj,
                ref=ref,
            )
            ani = FuncAnimation(
                self.fig, partial_animate_step, frames=n_steps, repeat=False, interval=interval, blit=True
            )

            print("💾 Saving video to disk...")
            ani.save(self.video_path, writer="ffmpeg")
            print(f"✅ Video saved successfully to {self.video_path}")

        else:
            print(f"🖼️ Generating {n_steps} frames in memory...")
            # FIXME: blit = True is 2-3x faster than blit = False, but for now the axis ticks will not update correctly
            blit = True
            if blit:
                self.fig.canvas.draw()
                self.ax_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            for i in range(n_steps):
                # Print progress every 10% of frames or every 10 frames, whichever is smaller
                progress_interval = max(1, min(10, n_steps // 10))
                if i % progress_interval == 0 or i == n_steps - 1:
                    progress_pct = (i + 1) / n_steps * 100
                    print(f"    📊 Frame {i + 1}/{n_steps} ({progress_pct:.1f}%)")
                
                state = state.replace(timestep=i)
                obs_t = operations.dynamic_slice(obs, i, 1, axis=0) if obs is not None else None
                self.plot_simulator_state(state, obs_t, use_log_traj, highlight_obj, ref)
                img = self.img_from_fig(close_fig=False, clear_fig=False, blit=blit)
                imgs.append(img)
            
            print("✅ All frames generated successfully in memory")

        self.close_fig()

        return imgs

    def plot_simulator_state(
        self,
        state: datatypes.SimulatorState,
        obs: GokartObs | None = None,
        use_log_traj: bool = True,
        highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
        ref: bool = False,
    ) -> None:
        """Plots np array image for SimulatorState.

        Args:
          state: A SimulatorState instance.
          obs: Optional observation instance.
          use_log_traj: Set True to use logged trajectory, o/w uses simulated
            trajectory.
          viz_config: dict for optional config.
          batch_idx: optional batch index.
          highlight_obj: Represents the type of objects that will be highlighted with
            `color.COLOR_DICT['controlled']` color.

        Returns:
          np image.
        """

        if state.shape:
            raise ValueError(f"Expecting 0 batch dimension, got {len(state.shape)}")

        # 1. Plots trajectory.
        traj = state.log_trajectory if use_log_traj else state.sim_trajectory

        # print("DEBUG: Shape of trajectory:", traj.shape)  # (4, 1000)
        # print("DEBUG: Object types:", state.object_metadata.object_types)
        # print("DEBUG: is_sdc:", state.object_metadata.is_sdc)
        # print("DEBUG: is_controlled:", state.object_metadata.is_controlled)
        # print("DEBUG: Object positions at current timestep:")
        # for i in range(traj.num_objects):
        #     print(f"  Object {i}: pos=({traj.x[i, state.timestep]:.3f}, {traj.y[i, state.timestep]:.3f}), "
        #           f"yaw={traj.yaw[i, state.timestep]:.3f}, valid={traj.valid[i, state.timestep]}")

        indices = np.arange(traj.num_objects) if self.viz_config.show_agent_id else None
        is_controlled = datatypes.get_control_mask(state.object_metadata, highlight_obj)
        
        # print("DEBUG: is_controlled mask:", is_controlled)
        
        self.plot_trajectory(
            traj, is_controlled, time_idx=state.timestep, indices=indices
        )  # pytype: disable=wrong-arg-types  # jax-ndarray

        if ref:
            ref_traj = state.log_trajectory
            traj_5dof = np.array(ref_traj.stack_fields(["x", "y", "length", "width", "yaw"]))  # Forces to np from jnp

            valid_controlled = is_controlled[:, np.newaxis] & ref_traj.valid
            if self.reference_lines is not None:
                self.reference_lines = self.ax.plot(
                    traj_5dof[valid_controlled][::5, 0],
                    traj_5dof[valid_controlled][::5, 1],
                    "-",
                    color=np.array([0.0, 0.0, 1.0]),
                    ms=1,
                    alpha=0.5,
                )
            else:
                self.reference_lines[0].set_data(
                    traj_5dof[valid_controlled][::5, 0],
                    traj_5dof[valid_controlled][::5, 1],
                )

        if obs is not None:
            self.plot_obs(traj=traj, timestep=state.timestep, obs=obs)

        # 2. Plots road graph elements.
        # assume roadgraph points do not change over time. Plot only once.
        if self.roadgraph_lines_dict is None:
            self.plot_roadgraph_points(state.roadgraph_points, verbose=False)

        if not self.skip_traffic_light:
            viz.plot_traffic_light_signals_as_points(self.ax, state.log_traffic_light, state.timestep, verbose=False)

        # 3. Gets img centered on selected agent's current location.
        # [A, 2]
        if self.faster_axis_origin:
            # 2x faster but doesn't do any checks
            origin_x = traj.x[0, state.timestep]
            origin_y = traj.y[0, state.timestep]
        else:
            current_xy = traj.xy[:, state.timestep, :]
            if self.viz_config.center_agent_idx == -1:
                xy = current_xy[state.object_metadata.is_sdc]
            else:
                xy = current_xy[self.viz_config.center_agent_idx]
            origin_x, origin_y = xy[0, :2]

        self.ax.axis(
            (
                origin_x - self.viz_config.back_x,
                origin_x + self.viz_config.front_x,
                origin_y - self.viz_config.back_y,
                origin_y + self.viz_config.front_y,
            )
        )

    def plot_obs(
        self,
        traj: datatypes.Trajectory,
        timestep: int,
        obs: GokartObs,
    ) -> None:

        # plot rays
        position = traj.xy[0, timestep, :]
        yaw = traj.yaw[0, timestep]
        # rays_length[0, timestep, :]
        self.plot_numpy_rays(
            position,
            yaw,
            color=np.array([1.0, 0.65, 0.0]),
            rays_length=obs.distances.squeeze(),
            rays_angles_base=obs._angles_rays,
            alpha=0.9,
        )

    def plot_trajectory(
        self,
        traj: datatypes.Trajectory,
        is_controlled: np.ndarray,
        time_idx: Optional[int] = None,
        indices: Optional[np.ndarray] = None,
        add_label: bool = False,
    ) -> None:
        """Plots a Trajectory with different color for controlled and context.

        Plots the full bounding_boxes only for time_idx step, overlap is
        highlighted.

        Notation: A: number of agents; T: numbe of time steps; 5 degree of freedom:
        center x, center y, length, width, yaw.

        Args:
          ax: matplotlib axes.
          traj: a Trajectory with shape (A, T).
          is_controlled: binary mask for controlled object, shape (A,).
          time_idx: step index to highlight bbox, -1 for last step. Default(None) for
            not showing bbox.
          indices: ids to show for each agents if not None, shape (A,).
          add_label: a boolean that indicates whether or not to plot labels that
            indicates different agent types, including 'controlled', 'overlap',
            'history', 'context'.
        """
        if len(traj.shape) != 2:
            raise ValueError("traj should have shape (A, T)")

        traj_5dof = np.array(traj.stack_fields(["x", "y", "length", "width", "yaw"]))  # Forces to np from jnp

        num_obj, num_steps, _ = traj_5dof.shape
        if time_idx is not None:
            if time_idx == -1:
                time_idx = num_steps - 1
            if time_idx >= num_steps:
                raise ValueError("time_idx is out of range.")

        # Adds id if needed.
        if indices is not None and time_idx is not None:
            for i in range(num_obj):
                if not traj.valid[i, time_idx]:
                    continue
                if self.text is not None:
                    if num_obj != 1:
                        self.text.set_text(f"{indices[i]}")
                    self.text.set_position((traj_5dof[i, time_idx, 0] - 2, traj_5dof[i, time_idx, 1] + 2))
                else:
                    self.text = self.ax.text(
                        traj_5dof[i, time_idx, 0] - 2,
                        traj_5dof[i, time_idx, 1] + 2,
                        f"{indices[i]}",
                        zorder=10,
                    )
        self._plot_bounding_boxes(
            traj_5dof=traj_5dof,
            time_idx=time_idx,
            is_controlled=is_controlled,
            valid=traj.valid,
            add_label=add_label,
        )  # pytype: disable=wrong-arg-types  # jax-ndarray

    def _plot_bounding_boxes(
        self,
        traj_5dof: np.ndarray,
        time_idx: int,
        is_controlled: np.ndarray,
        valid: np.ndarray,
        add_label: bool = False,
        controlled_next_steps_as_center_pts: bool = True,
    ) -> None:
        """Helper function to plot multiple bounding boxes across time."""
        # Plots bounding boxes (traj_5dof) with shape: (A, T)
        # is_controlled: (A,)
        # valid: (A, T)
        valid_controlled = is_controlled[:, np.newaxis] & valid
        valid_context = ~is_controlled[:, np.newaxis] & valid

        num_obj = traj_5dof.shape[0]
        time_indices = np.tile(np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1))
    
        # Shrinks bounding_boxes for non-current steps.
        traj_5dof[time_indices != time_idx, 2:4] /= 10
        
        # Check what will be plotted in each category
        controlled_mask = (time_indices >= time_idx) & valid_controlled
        # Para objetos de contexto (obstáculos estáticos), solo mostrar el timestep actual
        context_mask = (time_indices == time_idx) & valid_context  # SOLO timestep actual, no historial
        
        controlled_bboxes = traj_5dof[controlled_mask]
        context_bboxes = traj_5dof[context_mask]
        
        self.plot_numpy_bounding_boxes(
            self.name_trajectory_lines,
            bboxes=controlled_bboxes,
            color=color.COLOR_DICT["controlled"],
            as_center_pts=controlled_next_steps_as_center_pts,
            center_pts_from_idx=1 if controlled_next_steps_as_center_pts else 0,
            label="controlled" if add_label else None,
        )

        if self.history_lines is None or not self.plot_last_history_only:
            self.plot_numpy_bounding_boxes(
                self.name_history_lines,
                bboxes=(
                    traj_5dof[valid] if self.plot_last_history_only else traj_5dof[(time_indices < time_idx) & valid]
                ),
                color=color.COLOR_DICT["history"],
                as_center_pts=True,
                label="history" if add_label else None,
            )

        self.plot_numpy_bounding_boxes(
            self.name_context_lines,
            bboxes=context_bboxes,
            color=color.COLOR_DICT["context"],
            label="context" if add_label else None,
        )

        # Shows current overlap
        # (A, A)
        overlap_fn = jax.jit(geometry.compute_pairwise_overlaps)
        overlap_mask_matrix = overlap_fn(traj_5dof[:, time_idx])
        # Remove overlap against invalid objects.
        overlap_mask_matrix = np.where(valid[None, :, time_idx], overlap_mask_matrix, False)
        # (A,)
        overlap_mask = np.any(overlap_mask_matrix, axis=1)

        # Get the overlapping bounding boxes.
        overlapping_bboxes = traj_5dof[:, time_idx][overlap_mask & valid[:, time_idx]]
        
        if overlapping_bboxes.shape[0] > 0:
            print(f"  🚨 COLLISION at timestep {time_idx}! {overlapping_bboxes.shape[0]} overlapping objects")
            for i, bbox in enumerate(overlapping_bboxes):
                print(f"    Collision bbox {i}: pos=({bbox[0]:.3f}, {bbox[1]:.3f})")

        self.plot_numpy_bounding_boxes(
            self.name_overlap_lines,
            bboxes=overlapping_bboxes,
            color=color.COLOR_DICT["overlap"],
            label="overlap" if add_label else None,
        )

    def plot_numpy_bounding_boxes(
        self,
        line_name: str,
        bboxes: np.ndarray,
        color: np.ndarray,
        alpha: Optional[float] = 1.0,
        as_center_pts: bool = False,
        center_pts_from_idx: int = 0,
        label: Optional[str] = None,
    ) -> None:
        """Plots multiple bounding boxes.

        Args:
          ax: Fig handles.
          bboxes: Shape (num_bbox, 5), with last dimension as (x, y, length, width,
            yaw).
          color: Shape (3,), represents RGB color for drawing.
          alpha: Alpha value for drawing, i.e. 0 means fully transparent.
          as_center_pts: If set to True, bboxes will be drawn as center points,
            instead of full bboxes.
          center_pts_from_idx: If as_center_pts is True, bboxes will be drawn as center points from this index,
            while previous indices will be drawn as full bboxes.
          label: String, represents the meaning of the color for different boxes.
        """
        # DEBUG: Check what we're trying to plot (solo para overlaps)
        if line_name == "overlap_lines" and bboxes.shape[0] > 0:
            print(f"🎨 COLLISION! Plotting {bboxes.shape[0]} {line_name} bboxes in color {color}")
        
        lines = getattr(self, line_name)
        if bboxes.ndim != 2 or bboxes.shape[1] != 5 or color.shape != (3,):
            raise ValueError(
                (
                    "Expect bboxes rank 2, last dimension of bbox 5, color of size 3," " got{}, {}, {} respectively"
                ).format(bboxes.ndim, bboxes.shape[1], color.shape)
            )

        if bboxes.shape[0] == 0:
            return  # Sin mensaje para casos normales sin bboxes

        if as_center_pts and center_pts_from_idx <= 0:
            if lines is not None:
                lines[0].set_data(bboxes[:, 0], bboxes[:, 1])
            else:
                lines = self.ax.plot(
                    bboxes[:, 0],
                    bboxes[:, 1],
                    "o",
                    color=color,
                    ms=2,
                    alpha=alpha,
                    label=label,
                )
        else:
            if as_center_pts:
                center_bboxes = bboxes[center_pts_from_idx:]
                bboxes = bboxes[:center_pts_from_idx]
            else:
                center_bboxes = None

            c = np.cos(bboxes[:, 4])
            s = np.sin(bboxes[:, 4])
            pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
            length, width = bboxes[:, 2], bboxes[:, 3]
            u = np.array((c, s))
            ut = np.array((s, -c))

            # Compute box corner coordinates.
            tl = pt + length / 2 * u - width / 2 * ut
            tr = pt + length / 2 * u + width / 2 * ut
            br = pt - length / 2 * u + width / 2 * ut
            bl = pt - length / 2 * u - width / 2 * ut

            # Compute heading arrow using center left/right/front.
            cl = pt - width / 2 * ut
            cr = pt + width / 2 * ut
            cf = pt + length / 2 * u

            plot_bboxes_x = [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :], cl[0, :], cr[0, :], cf[0, :], cl[0, :]]
            plot_bboxes_y = [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :], cl[1, :], cr[1, :], cf[1, :], cl[1, :]]

            # Draw bboxes and heading arrow.
            if center_bboxes is not None:
                if lines is not None:
                    lines[0].set_data(plot_bboxes_x, plot_bboxes_y)
                    lines[1].set_data(center_bboxes[:, 0], center_bboxes[:, 1])
                else:
                    lines = self.ax.plot(
                        plot_bboxes_x,
                        plot_bboxes_y,
                        "-",
                        center_bboxes[:, 0],
                        center_bboxes[:, 1],
                        "o",
                        color=color,
                        ms=2,
                        zorder=4,
                        alpha=alpha,
                        label=label,
                    )
            else:
                if lines is not None:
                    lines[0].set_data(plot_bboxes_x, plot_bboxes_y)
                else:
                    lines = self.ax.plot(
                        plot_bboxes_x,
                        plot_bboxes_y,
                        color=color,
                        zorder=10,  # Más alto que el default (4) para que aparezca encima
                        alpha=alpha,
                        linewidth=2.0 if line_name == "context_lines" else 1.0,  # Líneas más gruesas para contexto
                        label=label,
                    )

        setattr(self, line_name, lines)

    def plot_numpy_rays(
        self,
        position: np.ndarray,
        yaw: np.ndarray,
        color: np.ndarray,
        rays_length: np.ndarray,
        rays_angles_base: np.ndarray,
        alpha: Optional[float] = 1.0,
    ) -> None:
        """
        Plots rays originating from a given position and orientation.

        Args:
            position: Array of shape (2,), representing the start position (x, y) of the rays.
            yaw: Array of shape (1,), representing the orientation angle of the source in radians.
            color: Array of shape (3,), representing the RGB color for the rays.
            rays_length: Array of shape (num_rays,), representing the length of each ray.
            rays_angles_base: Array of shape (num_rays,), representing the angles of the rays
                in radians, relative to the sensor base.
            num_rays: Number of rays to cast in the range of [-pi/2, pi/2] relative to the orientation.
            alpha: Alpha value for drawing, where 0 is fully transparent.
        """

        # Calculate angles for each ray relative to the orientation of the source.
        angles = rays_angles_base + yaw
        rays = position[:, None] + rays_length * np.array([np.cos(angles), np.sin(angles)])
        plot_xy = np.empty((2, rays.shape[1] * 2))
        plot_xy[:, ::2] = position[:, None]
        plot_xy[:, 1::2] = rays

        if self.rays_lines is not None:
            self.rays_lines[0].set_data(plot_xy[0], plot_xy[1])
        else:
            self.rays_lines = self.ax.plot(
                plot_xy[0],
                plot_xy[1],
                ":",
                color=color,
                alpha=alpha,
                zorder=4,
                linewidth=1.0,
            )

    def plot_roadgraph_points(
        self,
        rg_pts: datatypes.RoadgraphPoints,
        verbose: bool = False,
    ) -> None:
        """Plots road graph as points.

        Args:
          ax: matplotlib axes.
          rg_pts: a RoadgraphPoints with shape (1,)
          verbose: print roadgraph points count if set to True.
        """
        if len(rg_pts.shape) != 1:
            raise ValueError(f"Roadgraph should be rank 1, got {len(rg_pts.shape)}")
        if rg_pts.valid.sum() == 0:
            return
        elif verbose:
            print(f"Roadgraph points count: {rg_pts.valid.sum()}")

        xy = rg_pts.xy[rg_pts.valid]
        rg_type = rg_pts.types[rg_pts.valid]
        for curr_type in np.unique(rg_type):
            if curr_type in viz._RoadGraphShown:
                p1 = xy[rg_type == curr_type]
                rg_color = color.ROAD_GRAPH_COLORS.get(curr_type, viz._RoadGraphDefaultColor)
                if self.roadgraph_lines_dict is None or curr_type not in self.roadgraph_lines_dict:
                    if self.roadgraph_lines_dict is None:
                        self.roadgraph_lines_dict = {}
                    self.roadgraph_lines_dict[curr_type] = self.ax.plot(p1[:, 0], p1[:, 1], ".", color=rg_color, ms=2)
                else:
                    self.roadgraph_lines_dict[curr_type].set_data(p1[:, 0], p1[:, 1])

    def img_from_fig(self, close_fig: bool = True, clear_fig: bool = False, blit: bool = True) -> np.ndarray:
        """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_rgb()."""
        # Just enough margin in the figure to display xticks and yticks.
        if blit:
            self.fig.canvas.restore_region(self.ax_background)
            if self.roadgraph_lines_dict is not None:
                for road_type in self.roadgraph_lines_dict:
                    self.ax.draw_artist(self.roadgraph_lines_dict[road_type][0])
            for lines in [
                self.history_lines,
                self.trajectory_lines,
                self.context_lines,
                self.overlap_lines,
                self.reference_lines,
                self.rays_lines,
            ]:
                if lines is not None:
                    for line in lines:
                        self.ax.draw_artist(line)
            if self.text is not None:
                self.ax.draw_artist(self.text)
            self.fig.canvas.blit(self.ax.bbox)
        else:
            self.fig.canvas.draw()
        data = np.array(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = data.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        if clear_fig:
            self.ax.cla()
        if close_fig:
            plt.close(self.fig)
        return img[:, :, :3]  # Convert RGBA to RGB
