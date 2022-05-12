

import jax
import jax.numpy as jnp
import quantecon as qe 
import timeit
import json
import argparse
from jax.config import config
import os
import warnings
warnings.filterwarnings('ignore')
config.update("jax_log_compiles", 1)
# to suppress watnings uncomment next two lines


# MANUAL VECTORIZATION
def get_T_manualvec(model: dict):
    params = model["params"]
    a = model["batched_grids"]["a"]
    y = model["batched_grids"]["y"]
    ap = model["batched_grids"]["ap"]
    P = model["batched_grids"]["P"]
  
    def u(c):
        return c**(1-params["gamma"]) / (1-params["gamma"])

    def T_manualvec(v):
        vp = jnp.dot(v, P)  # vp has shape (a_size, y_size, 1)
        c = params["R"] * a + y - ap  # c has shape (a_size, y_size, ap_size)
        # m = jnp.where(c > 0, u(c) + β * vp, -np.inf) # m has shape (a_size, y_size, ap_size)
        m = u(c) + params["beta"] * vp  # m has shape (a_size, y_size, ap_size)
        return jnp.max(m, axis=2)  # we to average over the last axis.
    
    return T_manualvec


# AUTOMATIMC VECTORIZATION
def get_T_autovec(model: dict):
    params = model["params"]
    grids = model["grids"]
    P = model["Trans_matrix"]
  
    def u(c):
        return c**(1-params["gamma"]) / (1-params["gamma"])

    def T_autovec(v: jnp.ndarray):

        def action_v(a_ind: int, y_ind: int, ap_ind: int, v: jnp.array):
            c = params["R"]*grids["a"][a_ind]+grids["y"][y_ind]-grids["ap"][ap_ind]
            return jnp.where(c > 0, c**(1-params["gamma"]) / (1-params["gamma"]) + params["beta"] * jnp.dot(v[ap_ind, :],P[y_ind, :]), - jnp.inf)
    # first vmap to calculate action value for all possible ap's.

        vmapped_action_v = jax.vmap(action_v, in_axes=(None, None, 0, None))

        # get the maximum of all action_values for a pair of (a,y)
        def one_state_v(a_ind: int, y_ind: int, ap_grid: jnp.array, v: jnp.array):
            return jnp.max(vmapped_action_v(a_ind, y_ind, ap_grid, v))

        # do vmaps over the other two dimensions.
        all_state_v = jax.vmap(jax.vmap(one_state_v, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))
        # calculate value fuction matrix
        a_indices = model["indices"]["a"]
        y_indices = model["indices"]["y"]
        ap_indices = model["indices"]["ap"]
        new_state_value = all_state_v(a_indices, y_indices, ap_indices, v)
        return new_state_value

    return T_autovec


# TPU PARALLELIZATION
def get_T_tpu(model: dict):

    params = model["params"]
    grids = model["grids"]
    P = model["Trans_matrix"]
  
    def u(c):
        return c**(1-params["gamma"]) / (1-params["gamma"])

    def T_tpu(a_partition: jnp.array, v: jnp.ndarray):

        def action_v(a_ind: int, y_ind: int, ap_ind: int, v: jnp.array):
            c = params["R"]*grids["a"][a_ind]+grids["y"][y_ind]-grids["ap"][ap_ind]
            return jnp.where(c > 0, c**(1-params["gamma"]) / (1-params["gamma"]) + params["beta"] * jnp.dot(v[ap_ind, :], P[y_ind, :]), - jnp.inf)
        # first vmap to calculate action value for all possible ap's.
        vmapped_action_v = jax.vmap(action_v, in_axes=(None, None, 0, None))

        # get the maximum of all action_values for a pair of (a,y)
        def one_state_v(a_ind: int, y_ind: int, ap_grid: jnp.array, v: jnp.array):
            return jnp.max(vmapped_action_v(a_ind, y_ind, ap_grid, v))

    # do vmaps over the other two dimensions.
        all_state_v = jax.vmap(jax.vmap(one_state_v, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))
        # calculate value fuction matrix
        a_indices = a_partition
        y_indices = model["indices"]["y"]
        ap_indices = model["indices"]["ap"]
        new_state_value = all_state_v(a_indices, y_indices, ap_indices, v)
        return new_state_value

    return T_tpu

# BENCHMARKING


def main(output_file=os.get_cwd()+"benchmark_output.json", use_TPU=True):

    if use_TPU:
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()

    # Global Parameteres
    params = {
        "R": 1.1,
        "beta": 0.99,
        "gamma": 2.5
    }
    a_min, a_max = 0.01, 2
    ρ = 0.9
    σ = 0.1
    n_devices = jax.local_device_count()

    # Loop over scale levels
    results_dict = {"Size of grid": [], "Manual Vectorization": [], "Automatic Vectorization": [], "TPU Parallelization": []}
    for scale in [1, 8, 16, 32]:
        # grid for assets
        a_size = ap_size = 1024*scale
        a_grid = jnp.linspace(a_min, a_max, a_size)  # grid for a
        ap_grid = jnp.linspace(a_min, a_max, a_size)  # grid for a'
        # grid for shocks
        y_size = 128*scale
        mc = qe.tauchen(ρ, σ, n=y_size)
        y_grid = jnp.exp(mc.state_values)
        P = jnp.array(mc.P)
        results_dict["Size of grid"].append(a_size*y_size)

        grids = {
            "a": a_grid,
            "y": y_grid,
            "ap": ap_grid,
            }

        batched_grids = {
            "P": jnp.reshape(P, (y_size, y_size, 1)),
            "a": jnp.reshape(a_grid, (a_size, 1, 1)),
            "y": jnp.reshape(y_grid, (1, y_size, 1)),
            "ap": jnp.reshape(ap_grid, (1, 1, ap_size)),
            } 

        model = {
                "params": params,
                "grids": grids,
                "batched_grids": batched_grids,
                "Trans_matrix": P,
                "indices": {"a": jnp.array(range(a_size)), "y": jnp.array(range(y_size)), "ap": jnp.array(range(ap_size))}
                }

        # initial value
        global v_init
        v_init = jnp.zeros((a_size, y_size))

        # Get and compile T functions
        T_manualvec = get_T_manualvec(model)
        global T_manualvec_jit
        T_manualvec_jit = jax.jit(T_manualvec).lower(v_init).compile()
        T_autovec = get_T_autovec(model)
        global T_autovec_jit
        T_autovec_jit = jax.jit(T_autovec).lower(v_init).compile()

        # run for 10 times and time it using timeit
        if use_TPU:
            a_partitions = jnp.reshape(model["indices"]["a"], (n_devices, a_size//n_devices))
            T_tpu = get_T_tpu(model)
            global T_tpu_jit
            T_tpu_jit = jax.pmap(T_tpu, in_axes=(0,None)).lower(a_partitions, v_init).compile()
        results_dict["Manual Vectorization"].append(timeit.timeit('T_manualvec_jit(v_init).block_until_ready()', globals=globals(), number=10)/10)
        results_dict["Automatic Vectorization"].append(timeit.timeit('T_autovec_jit(v_init).block_until_ready()', globals=globals(), number=10)/10)
        results_dict["TPU Parallelization"].append(timeit.timeit('T_tpu_jit(a_partitions, v_init).block_until_ready()', globals=globals(), number=10)/10)

        with open(output_file, "w") as outfile:
            json.dump(results_dict, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the time required to run an update of the value function for different scales and methods.')
    parser.add_argument('-o', '--output', help='Output file (json)', required=True, type=str) # Output file arg
    args = vars(parser.parse_args())
    outfile = args['output']
    main(outfile)
