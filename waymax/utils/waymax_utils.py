from jax import numpy as jnp
from waymax import datatypes
from chex import dataclass
from dataclasses import fields

def replicate_init_state_to_form_batch(init_state: datatypes.SimulatorState, batch_size: int):
    '''
    replicate a SimulatorState multiple times to constrauct a batch
    '''
    assert len(init_state.shape) == 0

    temp_sim_trajectory = init_state.sim_trajectory
    temp_log_trajectory = init_state.log_trajectory
    temp_log_traffic_light = init_state.log_traffic_light
    temp_object_metadata = init_state.object_metadata
    temp_timestep = init_state.timestep
    assert init_state.sdc_paths is None
    temp_roadgraph_points = init_state.roadgraph_points

    def replicate_attr_in_class_sample(class_sample: dataclass, batch_size: int = batch_size):
        attr_names = [field.name for field in fields(class_sample)]
        for i in range(len(attr_names)):
            setattr(class_sample,attr_names[i],jnp.expand_dims(getattr(class_sample,attr_names[i]),0).repeat(batch_size,axis=0))
        return class_sample

    batched_init_states = datatypes.SimulatorState(
        sim_trajectory = replicate_attr_in_class_sample(temp_sim_trajectory),
        log_trajectory = replicate_attr_in_class_sample(temp_log_trajectory),
        log_traffic_light = replicate_attr_in_class_sample(temp_log_traffic_light),
        object_metadata = replicate_attr_in_class_sample(temp_object_metadata),
        timestep = jnp.expand_dims(temp_timestep,0).repeat(batch_size),
        sdc_paths = None,
        roadgraph_points = replicate_attr_in_class_sample(temp_roadgraph_points),
    )
    assert batched_init_states.shape[0] == batch_size
    return batched_init_states