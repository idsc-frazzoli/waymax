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

"""Utility function that runs all metrics according to an environment config."""
from collections.abc import Iterable
from typing import Any, Dict

from waymax import config as _config
from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.metrics import comfort
from waymax.metrics import imitation
from waymax.metrics import overlap
from waymax.metrics import roadgraph
from waymax.metrics import route
from waymax.metrics import gokart_progress
from waymax.metrics import gokart_offroad
from waymax.metrics import gokart_orientation
from waymax.metrics import gokart_action
from waymax.metrics import gokart_state


_METRICS_DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    "gokart_offroad": {"safety_margin": 0.0},
    "gokart_distance_to_bounds": {"safety_margin": 0.3, "additional_offroad_reward": 0},
    "gokart_vel_x": {"state_names": "vel_x"},
    "gokart_vel_y": {"state_names": "vel_y"},
    "gokart_vel_xy": {"state_names": ["vel_x", "vel_y"]},
    "gokart_yaw_rate": {"state_names": "yaw_rate"},
    "gokart_vel_x_out_range": {"state_names": "vel_x", "min_value": -2.0, "max_value": 6.0},
    "gokart_vel_y_out_range": {"state_names": "vel_y", "min_value": -1.5, "max_value": 1.5},
    "gokart_steer_action": {"action_names": "steering_angle"},
    "gokart_throttle_action": {"action_names": ["acc_left", "acc_right"]},
    "gokart_steer_action_rate": {"action_names": "steering_angle"},
    "gokart_throttle_action_rate": {"action_names": ["acc_left", "acc_right"]},
}

_METRICS_REGISTRY: dict[str, abstract_metric.AbstractMetric] = {
    "log_divergence": imitation.LogDivergenceMetric(),
    "overlap": overlap.OverlapMetric(),
    "offroad": roadgraph.OffroadMetric(),
    "kinematic_infeasibility": comfort.KinematicsInfeasibilityMetric(),
    "sdc_wrongway": roadgraph.WrongWayMetric(),
    "sdc_progression": route.ProgressionMetric(),
    "sdc_off_route": route.OffRouteMetric(),
    "gokart_progress": gokart_progress.GokartProgressMetric(),
    "gokart_orientation": gokart_orientation.GokartOrientationMetric(),
    "gokart_offroad": gokart_offroad.GokartOffroadMetric(**_METRICS_DEFAULT_ARGS["gokart_offroad"]),
    "gokart_distance_to_bounds": gokart_offroad.GokartDistanceToBoundsMetric(**_METRICS_DEFAULT_ARGS["gokart_distance_to_bounds"]),
    "gokart_vel_x": gokart_state.GokartStateNormMetric(**_METRICS_DEFAULT_ARGS["gokart_vel_x"]),
    "gokart_vel_y": gokart_state.GokartStateNormMetric(**_METRICS_DEFAULT_ARGS["gokart_vel_y"]),
    "gokart_vel_xy": gokart_state.GokartStateNormMetric(**_METRICS_DEFAULT_ARGS["gokart_vel_xy"]),
    "gokart_yaw_rate": gokart_state.GokartStateNormMetric(**_METRICS_DEFAULT_ARGS["gokart_yaw_rate"]),
    "gokart_vel_x_out_range": gokart_state.GokartStateOutRangeMetric(**_METRICS_DEFAULT_ARGS["gokart_vel_x_out_range"]),
    "gokart_vel_y_out_range": gokart_state.GokartStateOutRangeMetric(**_METRICS_DEFAULT_ARGS["gokart_vel_y_out_range"]),
    "gokart_action": gokart_action.GokartActionNormMetric(),
    "gokart_steer_action": gokart_action.GokartActionNormMetric(**_METRICS_DEFAULT_ARGS["gokart_steer_action"]),
    "gokart_throttle_action": gokart_action.GokartActionNormMetric(**_METRICS_DEFAULT_ARGS["gokart_throttle_action"]),
    "gokart_tv_action": gokart_action.GokartTVActionNormMetric(),
    "gokart_action_rate": gokart_action.GokartActionRateNormMetric(),
    "gokart_steer_action_rate": gokart_action.GokartActionRateNormMetric(**_METRICS_DEFAULT_ARGS["gokart_steer_action_rate"]),
    "gokart_throttle_action_rate": gokart_action.GokartActionRateNormMetric(**_METRICS_DEFAULT_ARGS["gokart_throttle_action_rate"]),
}    

def run_metrics(
    simulator_state: datatypes.SimulatorState,
    metrics_config: _config.MetricsConfig,
) -> dict[str, abstract_metric.MetricResult]:
    """Runs all metrics with config flags set to True.

    User-defined metrics must be registered using the `register_metric` function.

    Args:
      simulator_state: The current simulator state of shape (...).
      metrics_config: Waymax metrics config.

    Returns:
      A dictionary of metric names mapping to metric result arrays where each
        metric is of shape (..., num_objects).
    """
    results = {}
    for metric_name in metrics_config.metrics_to_run:
        if metric_name in _METRICS_REGISTRY:
            results[metric_name] = _METRICS_REGISTRY[metric_name].compute(simulator_state)
        else:
            raise ValueError(f"Metric {metric_name} not registered.")

    return results


def register_metric(metric_name: str, metric: abstract_metric.AbstractMetric, exist_ok: bool = False):
    """Register a metric.

    This function registers a metric so that it can be included in a MetricsConfig
    and computed by `run_metrics`.

    Args:
      metric_name: String name to register the metric with.
      metric: The metric to register.
    """
    if metric_name in _METRICS_REGISTRY and not exist_ok:
        raise ValueError(f"Metric {metric_name} has already been registered.")
    _METRICS_REGISTRY[metric_name] = metric


def get_metric_names() -> Iterable[str]:
    """Returns the names of all registered metrics."""
    return _METRICS_REGISTRY.keys()


def get_metric(metric_name: str) -> abstract_metric.AbstractMetric:
    """Returns the type of a registered metric given the metric name."""
    if metric_name not in _METRICS_REGISTRY:
        raise ValueError(f"Metric {metric_name} not registered.")
    return _METRICS_REGISTRY[metric_name]


def update_metrics_registry(reward_args: Dict[str, Dict[str, Any]]):
    for metric_name, metric_args in reward_args.items():
        metric_class = get_metric(metric_name).__class__
        metric_default_args = _METRICS_DEFAULT_ARGS.get(metric_name, None)
        if metric_default_args is not None:
            metric_default_args.update(metric_args)
            metric_new_args = metric_default_args
        else:
            metric_new_args = metric_args
        register_metric(metric_name, metric_class(**metric_new_args), exist_ok=True)

