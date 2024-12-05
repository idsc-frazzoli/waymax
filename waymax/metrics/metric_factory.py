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
    "gokart_offroad": gokart_offroad.GokartOffroadMetric(),
    "gokart_almost_offroad": gokart_offroad.GokartOffroadMetric(0.2),
    "gokart_distance_to_bounds": gokart_offroad.GokartDistanceToBoundsMetric(0.3),
    "gokart_vel_x": gokart_state.GokartVelxKernelMetric(),
    "gokart_vel_y": gokart_state.GokartVelyKernelMetric(),
    "gokart_yaw_rate": gokart_state.GokartStateKernelMetric("yaw_rate"), # example of a custom metric
    "gokart_vel_x_out_range": gokart_state.GokartVelxOutRangeMetric(-2.0, 6.0),
    "gokart_vel_y_out_range": gokart_state.GokartStateOutRangeMetric("vel_y", -3.0, 3.0), # example of a custom metric
    "gokart_action": gokart_action.GokartActionMetric(),
    "gokart_steer_action": gokart_action.GokartActionMetric(["steering_angle"]),
    "gokart_throttle_action": gokart_action.GokartActionMetric(["AB_L", "AB_R"]),
    "gokart_tv_action": gokart_action.GokartActionTVMetric(),
    "gokart_action_rate": gokart_action.GokartActionRateMetric(),
    "gokart_steer_action_rate": gokart_action.GokartActionRateMetric(["steering_angle"]),
    "gokart_throttle_action_rate": gokart_action.GokartActionRateMetric(["AB_L", "AB_R"]),
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


def get_metric_type(metric_name: str) -> type[abstract_metric.AbstractMetric]:
    """Returns the type of a registered metric given the metric name."""
    if metric_name not in _METRICS_REGISTRY:
        raise ValueError(f"Metric {metric_name} not registered.")
    return _METRICS_REGISTRY[metric_name].__class__
