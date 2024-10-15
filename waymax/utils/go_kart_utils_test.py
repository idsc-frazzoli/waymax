import jax.numpy as jnp
import matplotlib.pyplot as plt
from waymax.utils.gokart_utils import create_batch_init_state, generate_racing_track,TrackControlPoints
from waymax import datatypes


track_cntrl_points = TrackControlPoints()
roadgraph_points, x_center, y_center, cumulative_length = generate_racing_track(
        track_cntrl_points.x,
        track_cntrl_points.y,
        track_cntrl_points.r)
edge_points = roadgraph_points.xy[2000:, :] # 2000 should be the number of points on the centerline
state = create_batch_init_state(batch_size=2)
is_road_edge = datatypes.is_road_edge(state.roadgraph_points.types)
is_road_edge = is_road_edge[0] # take the first batch
edge_points_test = state.roadgraph_points.xy[0]  # take the first batch
edge_points_test = edge_points_test[is_road_edge]
print(f"edge points num: {edge_points_test.shape[0]}")

assert jnp.array_equal(edge_points, edge_points_test) 

# state.current_sim_trajectory.xy  shape = (batch_size, num_objects, num_timesteps = 1, 2)
car_pos = state.current_sim_trajectory.xy[1, 0, 0, :] 
car_orientation = state.current_sim_trajectory.yaw[1, 0, 0]
plt.figure(figsize=(10, 6))

# plot car position and orientation
plt.scatter(*car_pos, color='red', label='Car Position')
car_direction = jnp.array([jnp.cos(car_orientation), jnp.sin(car_orientation)])
plt.arrow(car_pos[0], car_pos[1], car_direction[0], car_direction[1],
          color='green', head_width=0.2, length_includes_head=True, label='Car Direction')

# plot track boundary points
plt.scatter(edge_points[:, 0], edge_points[:, 1], color='blue', label='Boundary Points', s=2)

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('racing track')
plt.grid(True)
plt.axis('equal')
plt.show()