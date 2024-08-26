import jax
from jax import numpy as jnp
import numpy as np
import mediapy
from tqdm import tqdm
import dataclasses
import chex
import copy

from scipy.interpolate import splprep, splev
from waymax import config as _config
from waymax import metrics
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax import visualization
from waymax.agents import actor_core
from waymax.utils.gokart_config import TrackControlPoints


def generate_racing_track(x, y, r, num_points=1001, batch_size=None):
# Calculate tangent and normal vectors for each control point
    x_left = []
    y_left = []
    x_right = []
    y_right = []

    for i in range(len(x)):
        if i == 0:
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
        elif i == len(x) - 1:
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
        else:
            dx = x[i+1] - x[i-1]
            dy = y[i+1] - y[i-1]

        length = np.sqrt(dx**2 + dy**2)

        dx /= length
        dy /= length

        normal_x = -dy
        normal_y = dx

        width = r[i]  # Dynamic width based on r

        x_left.append(x[i] + normal_x * width / 2)
        y_left.append(y[i] + normal_y * width / 2)
        x_right.append(x[i] - normal_x * width / 2)
        y_right.append(y[i] - normal_y * width / 2)


    # Create new splines for left and right edges using the control points
    tck, _ = splprep([x, y], s=0, per=1)
    tck_left, _ = splprep([x_left, y_left], s=0, per=1)
    tck_right, _ = splprep([x_right, y_right], s=0, per=1)

    u_new = np.linspace(0, 1, num_points)
    x_center, y_center = splev(u_new, tck)
    x_left, y_left = splev(u_new, tck_left)
    x_right, y_right = splev(u_new, tck_right)

    # Remove the last point to avoid duplicate points
    x_center = x_center[:-1]
    y_center = y_center[:-1]
    x_left = x_left[:-1]
    y_left = y_left[:-1]
    x_right = x_right[:-1]
    y_right = y_right[:-1]


    dx_center = np.diff(x_center)
    dy_center = np.diff(y_center)

    segment_lengths = np.sqrt(dx_center**2 + dy_center**2)
    cumulative_length = np.cumsum(segment_lengths)
    cumulative_length = np.insert(cumulative_length, 0, 0)

    normalized_length = cumulative_length / cumulative_length[-1]

    dx_left = np.diff(x_left)
    dy_left = np.diff(y_left)
    dx_right = np.diff(x_right)
    dy_right = np.diff(y_right)
    
    # To keep the directions array the same length as the other arrays, append the last direction
    dx_center = np.append(dx_center, dx_center[-1])
    dy_center = np.append(dy_center, dy_center[-1])
    dx_left = np.append(dx_left, dx_left[-1])
    dy_left = np.append(dy_left, dy_left[-1])
    dx_right = np.append(dx_right, dx_right[-1])
    dy_right = np.append(dy_right, dy_right[-1])

    # normalize the directions
    dx_center /= np.sqrt(dx_center**2 + dy_center**2)
    dy_center /= np.sqrt(dx_center**2 + dy_center**2)
    dx_left /= np.sqrt(dx_left**2 + dy_left**2)
    dy_left /= np.sqrt(dx_left**2 + dy_left**2)
    dx_right /= np.sqrt(dx_right**2 + dy_right**2)
    dy_right /= np.sqrt(dx_right**2 + dy_right**2)

    x_points = np.concatenate([x_center, x_left, x_right])
    y_points = np.concatenate([y_center, y_left, y_right])
    dir_x = np.concatenate([dx_center, -dx_left, dx_right])   # different directions for left and right boundary ????
    dir_y = np.concatenate([dy_center, -dy_left, dy_right])
    point_types = np.ones(num_points-1, dtype=np.int32)
    point_types = np.concatenate([point_types, point_types*15, point_types*15]) # 1: centerline, 15: road boundary

    x_points = jnp.array(x_points, dtype=jnp.float32)
    y_points = jnp.array(y_points, dtype=jnp.float32)
    dir_x = jnp.array(dir_x, dtype=jnp.float32)
    dir_y = jnp.array(dir_y, dtype=jnp.float32)
    point_types = jnp.array(point_types, dtype=jnp.int32)
    if batch_size is None:
        roadgraph_points = datatypes.RoadgraphPoints(
            x=x_points, # shape (num_points-1,)
            y=y_points,
            z=jnp.zeros_like(x_points),
            dir_x=dir_x,
            dir_y=dir_y,
            dir_z=jnp.zeros_like(x_points),
            types=point_types,
            ids=jnp.ones_like(x_points, dtype=jnp.int32),   # different ids for left and right boundary ??
            valid=jnp.ones_like(x_points, dtype=jnp.bool_),
        )
    else:        
        x_points = jnp.expand_dims(x_points, axis=0).repeat(batch_size, axis=0) # shape (batch_size, num_points-1=1000)
        y_points = jnp.expand_dims(y_points, axis=0).repeat(batch_size, axis=0)
        dir_x = jnp.expand_dims(dir_x, axis=0).repeat(batch_size, axis=0)
        dir_y = jnp.expand_dims(dir_y, axis=0).repeat(batch_size, axis=0)
        point_types = jnp.expand_dims(point_types, axis=0).repeat(batch_size, axis=0)

        roadgraph_points = datatypes.RoadgraphPoints(
            x=x_points,   
            y=y_points,
            z=jnp.zeros_like(x_points),
            dir_x=dir_x,
            dir_y=dir_y,
            dir_z=jnp.zeros_like(x_points, dtype=jnp.float32),
            types=point_types,
            ids=jnp.ones_like(x_points, dtype=jnp.int32),   # different ids for left and right boundary ??
            valid=jnp.ones_like(x_points, dtype=jnp.bool_),
        )


    return roadgraph_points, jnp.array(x_center), jnp.array(y_center) ,cumulative_length

def create_init_state(num_timesteps = 300):
    '''
    create a GoKartSimState object with the generated track
    Since we don't have a log trajectory, we set the first point of the log trajectory to the first point of the centerline and
    the last point of the log trajectory to the last point of the centerline, which will be used to calculate the progress (see metric sdc_progression)
    We use the centerline as the sdc path (reference path) 
    '''
    trajectory = datatypes.GoKartTrajectory.zeros((1, num_timesteps))  # 1 object, 200 time steps
    sim_trajectory = trajectory
    sim_trajectory.length = jnp.ones_like(sim_trajectory.length) * 1.5
    sim_trajectory.width = jnp.ones_like(sim_trajectory.width)
    sim_trajectory.height = jnp.ones_like(sim_trajectory.height)
    sim_trajectory.valid = sim_trajectory.valid.at[0,0].set(True)
    # sim_trajectory.valid = jnp.ones_like(sim_trajectory.valid, dtype=jnp.bool_)
    sim_trajectory.x = sim_trajectory.x.at[0, 0].set(TrackControlPoints.x[0])
    sim_trajectory.y = sim_trajectory.y.at[0, 0].set(TrackControlPoints.y[0])

    # not used in go-kart simulation
    traffic_light = datatypes.TrafficLights(x=jnp.zeros((1, num_timesteps)), 
                                            y=jnp.zeros((1, num_timesteps)), 
                                            z = jnp.zeros((1, num_timesteps)), 
                                            state=jnp.zeros((1, num_timesteps), dtype=jnp.int32), 
                                            lane_ids=jnp.zeros((1, num_timesteps), dtype=jnp.int32), 
                                            valid=jnp.ones((1, num_timesteps), dtype=jnp.bool_))
    metadata = datatypes.ObjectMetadata(ids=jnp.array([0]), 
                                        object_types=jnp.array([1]), 
                                        is_sdc=jnp.array([True]), 
                                        is_modeled=jnp.array([False]), 
                                        is_valid=jnp.array([True]), 
                                        objects_of_interest=jnp.array([False]), 
                                        is_controlled=jnp.array([True]))
    timestep = jnp.array(0, dtype=jnp.int32)
    
    roadgraph_points, x_center, y_center, cumulative_length = generate_racing_track(TrackControlPoints.x, TrackControlPoints.y, TrackControlPoints.r)

    dx_path = jnp.diff(x_center)
    dy_path = jnp.diff(y_center)
    dx_path = jnp.append(dx_path, dx_path[-1])
    dy_path = jnp.append(dy_path, dy_path[-1])
    dx_path /= jnp.sqrt(dx_path**2 + dy_path**2)
    dy_path /= jnp.sqrt(dx_path**2 + dy_path**2)
    dx_path = jnp.expand_dims(dx_path, axis=-2)
    dy_path = jnp.expand_dims(dy_path, axis=-2)
    x_path = jnp.expand_dims(x_center, axis=-2)
    y_path = jnp.expand_dims(y_center, axis=-2)
    sdc_path = datatypes.GoKartPaths(
        x = x_path,
        y = y_path,
        z = jnp.zeros_like(x_path),
        dir_x = dx_path,
        dir_y = dy_path,
        ids = jnp.zeros_like(x_path, dtype=jnp.int32),
        valid=jnp.ones_like(x_path, dtype=jnp.bool_),
        arc_length= jnp.expand_dims(cumulative_length, axis=-2),
        on_route=jnp.ones((1, 1), dtype=jnp.bool_)
    )
    
    log_trajectory = copy.deepcopy(sim_trajectory)
    log_trajectory.x = log_trajectory.x.at[0, -1].set(x_center[-1])
    log_trajectory.y = log_trajectory.y.at[0, -1].set(y_center[-1])

    return datatypes.GoKartSimState(sim_trajectory=sim_trajectory,log_trajectory=trajectory, log_traffic_light=traffic_light, 
                                    object_metadata=metadata, timestep=timestep, roadgraph_points=roadgraph_points, sdc_paths=sdc_path)



def create_batch_init_state(batch_size = 2, num_timesteps = 200):
    '''
    create a GoKartSimState with batch_size
    '''
    sim_trajectory = datatypes.GoKartTrajectory.zeros((batch_size, 1, num_timesteps))  # 1 object, 200 time steps
    sim_trajectory.length = jnp.ones_like(sim_trajectory.length) * 1.5
    sim_trajectory.width = jnp.ones_like(sim_trajectory.width)
    sim_trajectory.height = jnp.ones_like(sim_trajectory.height)
    sim_trajectory.valid = sim_trajectory.valid.at[:,0,0].set(True)
    # sim_trajectory.valid = jnp.ones_like(sim_trajectory.valid, dtype=jnp.bool_)

    # set the first point of the centerline as the starting point of the sim_trajectory
    sim_trajectory.x = sim_trajectory.x.at[:, 0, 0].set(TrackControlPoints.x[0])
    sim_trajectory.y = sim_trajectory.y.at[:, 0, 0].set(TrackControlPoints.y[0])

    # not used in go-kart simulation
    traffic_light = datatypes.TrafficLights(x=jnp.zeros((batch_size, 1, num_timesteps)), 
                                            y=jnp.zeros((batch_size, 1, num_timesteps)), 
                                            z = jnp.zeros((batch_size, 1, num_timesteps)), 
                                            state=jnp.zeros((batch_size, 1, num_timesteps), dtype=jnp.int32), 
                                            lane_ids=jnp.zeros((batch_size, 1, num_timesteps), dtype=jnp.int32), 
                                            valid=jnp.ones((batch_size, 1, num_timesteps), dtype=jnp.bool_))
    metadata = datatypes.ObjectMetadata(ids=jnp.zeros((batch_size,1), dtype=jnp.int32), 
                                        object_types=jnp.ones((batch_size,1), dtype = jnp.int32), 
                                        is_sdc=jnp.ones((batch_size,1), dtype = jnp.bool_), 
                                        is_modeled=jnp.zeros((batch_size,1), dtype=jnp.bool_), 
                                        is_valid=jnp.ones((batch_size,1), dtype = jnp.bool_), 
                                        objects_of_interest=jnp.zeros((batch_size,1), dtype=jnp.bool_), 
                                        is_controlled=jnp.ones((batch_size,1), dtype = jnp.bool_))   # shape (batch_size, num_objects=1)
    timestep = jnp.zeros((batch_size,), dtype=jnp.int32)
    
    roadgraph_points, x_center, y_center, cumulative_length = generate_racing_track(TrackControlPoints.x, TrackControlPoints.y, TrackControlPoints.r, batch_size=batch_size)

    dx_path = jnp.diff(x_center)
    dy_path = jnp.diff(y_center)
    dx_path = jnp.append(dx_path, dx_path[-1])
    dy_path = jnp.append(dy_path, dy_path[-1])
    dx_path /= jnp.sqrt(dx_path**2 + dy_path**2)
    dy_path /= jnp.sqrt(dx_path**2 + dy_path**2)
    dx_path = jnp.expand_dims(dx_path, axis=(-2,-3))
    dy_path = jnp.expand_dims(dy_path, axis=(-2,-3))
    x_path = jnp.expand_dims(x_center, axis=(-2,-3)) 
    y_path = jnp.expand_dims(y_center, axis=(-2,-3))
    
    sdc_path = datatypes.GoKartPaths(
        x = x_path.repeat(batch_size, axis=0), # shape(batch_size, num_paths=1, num_points_per_path)
        y = y_path.repeat(batch_size, axis=0),
        z = jnp.zeros_like(x_path).repeat(batch_size, axis=0),
        dir_x = dx_path.repeat(batch_size, axis=0),
        dir_y = dy_path.repeat(batch_size, axis=0),
        ids = jnp.zeros_like(x_path, dtype=jnp.int32).repeat(batch_size, axis=0),
        valid=jnp.ones_like(x_path, dtype=jnp.bool_).repeat(batch_size, axis=0),
        arc_length= jnp.expand_dims(cumulative_length, axis=(-2,-3)).repeat(batch_size, axis=0),
        on_route=jnp.ones((batch_size, 1, 1), dtype=jnp.bool_) # shape (batch_size, num_paths=1, 1) 
    )

    log_trajectory = copy.deepcopy(sim_trajectory)
    # set the last point of the log trajectory to the last point of the centerline
    log_trajectory.x = log_trajectory.x.at[:, 0, -1].set(x_center[-1])
    log_trajectory.y = log_trajectory.y.at[:, 0, -1].set(y_center[-1])


    return datatypes.GoKartSimState(sim_trajectory=sim_trajectory,log_trajectory=log_trajectory, log_traffic_light=traffic_light, 
                                    object_metadata=metadata, timestep=timestep, roadgraph_points=roadgraph_points, sdc_paths=sdc_path)