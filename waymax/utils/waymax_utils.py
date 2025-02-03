import jax
from jax import numpy as jnp
from waymax import datatypes
from chex import dataclass
from dataclasses import fields

def replicate_init_state_to_form_batch(init_state: list[datatypes.SimulatorState], batch_size: int):
    '''
    replicate a SimulatorState multiple times to construct a batch
    '''
    num_env = len(init_state)
    assert batch_size%num_env == 0, "Currently, batch_size must be an interger multiple of num_env."
    rep = batch_size//num_env

    ### initial implementation

    # temp_sim_trajectory = [ele.sim_trajectory for ele in init_state]
    # temp_log_trajectory = [ele.log_trajectory for ele in init_state]
    # temp_log_traffic_light = [ele.log_traffic_light for ele in init_state]
    # temp_object_metadata = [ele.object_metadata for ele in init_state]
    # temp_timestep = [ele.timestep for ele in init_state]
    # # assert init_state.sdc_paths is None
    # temp_roadgraph_points = [ele.roadgraph_points for ele in init_state]

    # def replicate_attrs_in_class_samples(class_samples: list[dataclass], rep_times: int = rep):
    #     attr_names = [field.name for field in fields(class_samples[0])]
    #     for i in range(len(attr_names)):
    #         attr_value_list = [jnp.expand_dims(getattr(sample,attr_names[i]),0).repeat(rep,axis=0) for sample in class_samples]
    #         attr_value = jnp.concatenate(attr_value_list)
    #         setattr(class_samples[0], attr_names[i], attr_value)
    #     return class_samples[0]

    # batched_init_states = datatypes.SimulatorState(
    #     sim_trajectory = replicate_attrs_in_class_samples(temp_sim_trajectory),
    #     log_trajectory = replicate_attrs_in_class_samples(temp_log_trajectory),
    #     log_traffic_light = replicate_attrs_in_class_samples(temp_log_traffic_light),
    #     object_metadata = replicate_attrs_in_class_samples(temp_object_metadata),
    #     timestep = jnp.expand_dims(temp_timestep[0],0).repeat(batch_size),
    #     sdc_paths = None,
    #     roadgraph_points = replicate_attrs_in_class_samples(temp_roadgraph_points),
    # )
    # assert batched_init_states.shape[0] == batch_size

    ### implementation using jax.tree.map

    def _add_batch_dim(x):
        return jnp.expand_dims(x, axis=0).repeat(rep, axis=0)
    
    replicated_init_states = []
    for ele in init_state:
        repicated_ele = jax.tree.map(_add_batch_dim, ele)
        repicated_ele.validate()
        replicated_init_states.append(repicated_ele)
    
    batched_init_states = jax.tree.map(lambda *v: jnp.concatenate(v), *replicated_init_states)
    batched_init_states.validate()
    assert batched_init_states.shape[0] == batch_size

    return batched_init_states
