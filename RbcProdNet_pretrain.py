import jax
from jax import lax
from jax import random
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import optax
import timeit
from typing import Sequence
import matplotlib.pyplot as plt
import scipy.io as sio

# from google.colab import drive
# # Mount Google Drive
# drive.mount('/content/drive')
print(jax.devices())
""" This script solves an RBC model with production networks,
usign Neural Nets as function approximators of the policy.
We will start with some imports """


"""Now we import model parameters and a solution to a loglinearized version of the model (solved in Dynare).
We will use the Dynare solution as a benchmark for our model. """

# import model parameters
matlab_struct = sio.loadmat(
    "./RbcProdNet_data/ProdNetPert_July16.mat", simplify_cells=True
)
print(matlab_struct.keys())


""" Parameteres of the model """
Sigma_A = jnp.array(matlab_struct["modvcv"])  # variance-covariance of TFP shocks
n_sectors = Sigma_A.shape[0]  # number of sectors
rho_vec = jnp.array(matlab_struct["modrho"])  # parameter rho
dep_vec = jnp.array(matlab_struct["moddel"])  # parameter delta
ss_states = jnp.array(
    matlab_struct["ss_states"]
)  # steady state of state variables (in logs)
ss_IQPL = jnp.array(
    matlab_struct["ss_IQPL"]
)  # steady state of policy variables (in logs)


""" Matrices of dynare state-space representation  model """
A = matlab_struct["A"]
B = matlab_struct["B"]
C_IQPL = matlab_struct["C_IQPL"]
D_IQPL = matlab_struct["D_IQPL"]

""" Standard deviation of state variables and policy  variables (in logs) """
sd_states = jnp.array(matlab_struct["sd_states"])
sd_IQPL = jnp.array(matlab_struct["sd_IQPL"])

print("done importing parameteres and dynare state space representation")
""" Pre Train Environment """

""" We will start by creating a class that initialize and step forward our economic model """


class ProdNetRbc_pretrain:
    """A JAX implementation of an RBC model with Production Networks."""

    def __init__(
        self,
        rho_vec=rho_vec,
        dep_vec=dep_vec,
        Sigma_A=Sigma_A,
        ss_states=ss_states,
        ss_policy=ss_IQPL,
        sd_states=sd_states,
        A=A,
        B=B,
        C=C_IQPL,
        D=D_IQPL,
    ):
        self.rho_vec = rho_vec
        self.dep_vec = dep_vec
        self.n_sectors = rho_vec.shape[0]
        self.Sigma_A = Sigma_A
        self.ss_states = ss_states
        self.ss_policy = ss_policy
        self.sd_states = sd_states
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.num_actions = 3 * self.n_sectors + 1

    def initial_state(self, rng):
        e = jax.random.multivariate_normal(
            rng, jnp.zeros((self.n_sectors,)), self.Sigma_A
        )
        state_init = jnp.divide(jnp.dot(self.B, e), self.sd_states)
        return lax.stop_gradient(state_init)

    def step(self, state, shock):
        state_notnorm = jnp.multiply(state, self.sd_states)
        new_state_notnorm = jnp.dot(self.A, state_notnorm) + jnp.dot(self.B, shock)
        new_state = jnp.divide(new_state_notnorm, self.sd_states)
        obs = jnp.concatenate([state[: self.n_sectors], new_state[self.n_sectors :]])
        policy_devs = jnp.dot(self.C, state_notnorm) + jnp.dot(self.D, shock)
        IQPL = jnp.exp(self.ss_policy + policy_devs)
        L_agg = jnp.array([jnp.sum(IQPL[3 * self.n_sectors :])])
        policy_dynare = jnp.concatenate([IQPL[: 3 * self.n_sectors], L_agg])
        train_pair = (obs, policy_dynare)
        return lax.stop_gradient((new_state, train_pair))


"""
Neural Net

First, we use Flax to create the Neural Net,
Notice that we activate the last layer using Softplus
 to guarantee that we get possitive outputs
 """


class MLP_softplus(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
            x = nn.softplus(nn.Dense(self.features[-1])(x))
        return x


""" Class to time the runs"""


class TimeIt:
    def __init__(self, tag, steps=None):
        self.tag = tag
        self.steps = steps

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.steps:
            msg += ", FPS=%.2e" % (self.steps / self.elapsed_secs)
            print(msg)


"""
Learner Function.
No we define a function that defines the entire workflow of an epoque
"""


def get_learner_fn(env, nn_forward, opt_update, batch_size, epoque_iters):
    """It runs and epoque with learing.
    This is what the compiler reads and parallelize
    (the minimal unit of computation)."""

    def traj_loss_fn(nn_params, loss_rng, env_state):
        # shocks for the entire trajectory.
        shocks = jax.random.multivariate_normal(
            loss_rng, jnp.zeros((env.n_sectors,)), env.Sigma_A, shape=(batch_size,)
        )
        # apply period steps for each row shock in shocks.
        state_final, train_pairs = lax.scan(env.step, env_state, shocks)
        obs_vector, dynare_policy_vector = train_pairs
        nn_policy_vector = nn_forward(nn_params, obs_vector)
        traj_abs_loss = jnp.mean(
            jnp.abs(
                jnp.divide(nn_policy_vector, dynare_policy_vector)
                - jnp.ones_like(dynare_policy_vector)
            )
        )
        traj_loss = jnp.mean(
            jnp.square(
                jnp.divide(nn_policy_vector, dynare_policy_vector)
                - jnp.ones_like(dynare_policy_vector)
            )
        )

        return traj_loss, (
            state_final,
            jnp.array([traj_loss]),
            jnp.array([traj_abs_loss]),
        )

    def update_fn(nn_params, opt_state, rng, env_state, mean_loss, mean_abs_loss):
        """Compute a gradient update from a single trajectory."""
        rng, loss_rng = random.split(rng)
        grads, aux_info = jax.grad(  # compute gradient on a single trajectory.
            traj_loss_fn, has_aux=True
        )(nn_params, loss_rng, env_state)
        new_env_state, mean_loss, mean_abs_loss = aux_info
        grads = lax.pmean(
            grads, axis_name="j"
        )  # reduce mean (average grads) across cores.
        grads = lax.pmean(
            grads, axis_name="i"
        )  # reduce mean (average grads) across batch.
        updates, new_opt_state = opt_update(grads, opt_state)  # transform grads.
        new_params = optax.apply_updates(nn_params, updates)  # update parameters.
        return new_params, new_opt_state, rng, new_env_state, mean_loss, mean_abs_loss

    def learner_fn(params, opt_state, rngs, env_states, mean_loss, mean_abs_loss):
        """Vectorise and repeat the update."""
        batched_update_fn = jax.vmap(
            update_fn, axis_name="j"
        )  # vectorize across batch.

        def iterate_fn(_, val):  # repeat many times to avoid going back to Python.
            params, opt_state, rngs, env_states, mean_loss, mean_abs_loss = val
            return batched_update_fn(
                params, opt_state, rngs, env_states, mean_loss, mean_abs_loss
            )

        return lax.fori_loop(
            0,
            epoque_iters,
            iterate_fn,
            (params, opt_state, rngs, env_states, mean_loss, mean_abs_loss),
        )

    return learner_fn


"""
Define entire experiment
"""


def run_experiment(env, config):
    """Runs experiment."""
    cores_count = len(jax.devices())  # get available TPU cores.
    nn_policy = MLP_softplus(config["layers"] + [env.num_actions])
    optim = optax.adam(config["learning_rate"])  # define optimiser.

    rng, rng_e, rng_p = random.split(
        random.PRNGKey(config["seed"]), num=3
    )  # prng keys.
    dummy_obs = env.initial_state(rng_e)  # dummy for net init.
    params = nn_policy.init(rng_p, dummy_obs)  # initialise params.
    nn_forward = nn_policy.apply
    mean_loss = jnp.array([0.0])  # initialize loss
    mean_abs_loss = jnp.array([0.0])  # initialize loss
    opt_state = optim.init(params)  # initialise optimiser stats.
    learn = get_learner_fn(
        env, nn_forward, optim.update, config["batch_size"], config["epoque_iters"]
    )
    learn = jax.pmap(learn, axis_name="i")  # replicate over multiple cores.

    def broadcast(x):
        return jnp.broadcast_to(x, (cores_count, config["n_batches"]) + x.shape)

    params = jax.tree_map(broadcast, params)  # broadcast to cores and batch.
    opt_state = jax.tree_map(broadcast, opt_state)  # broadcast to cores and batch
    mean_loss = jax.tree_map(broadcast, mean_loss)
    mean_abs_loss = jax.tree_map(broadcast, mean_abs_loss)

    rng, *env_rngs = jax.random.split(rng, cores_count * config["n_batches"] + 1)
    env_states = jax.vmap(env.initial_state)(jnp.stack(env_rngs))  # init envs.
    rng, *step_rngs = jax.random.split(rng, cores_count * config["n_batches"] + 1)
    rng, *eval_rngs = jax.random.split(rng, cores_count * config["n_batches"] + 1)

    def reshape(x):
        return x.reshape((cores_count, config["n_batches"]) + x.shape[1:])

    step_rngs = reshape(jnp.stack(step_rngs))  # add dimension to pmap over.
    eval_rngs = reshape(jnp.stack(eval_rngs))  # add dimension to pmap over.
    env_states = reshape(env_states)  # add dimension to pmap over.

    mean_losses = []
    mean_accuracy = []
    num_steps = (
        cores_count
        * config["epoque_iters"]
        * config["batch_size"]
        * config["n_batches"]
    )

    with TimeIt(tag="COMPILATION"):
        learn(
            params, opt_state, step_rngs, env_states, mean_loss, mean_abs_loss
        )  # compiles

    # First run, we calculate periods per second
    with TimeIt(tag="EXECUTION", steps=num_steps):
        params, opt_state, step_rngs, env_states, mean_loss, mean_abs_loss = learn(
            params, opt_state, step_rngs, env_states, mean_loss, mean_abs_loss
        )

    # Rest of the runs
    for i in range(2, config["n_epoques"] + 1):
        rng, *step_rngs = jax.random.split(rng, cores_count * config["n_batches"] + 1)
        step_rngs = reshape(jnp.stack(step_rngs))
        params, opt_state, step_rngs, env_states, mean_loss, mean_abs_loss = learn(
            params, opt_state, step_rngs, env_states, mean_loss, mean_abs_loss
        )

        mean_losses.append(jnp.mean(mean_loss))
        mean_accuracy.append((1 - jnp.mean(mean_abs_loss)) * 100)

        print(
            "Iteration:",
            i * config["epoque_iters"],
            ", Mean_loss:",
            jnp.mean(mean_loss),
            ", Learning rate:",
            config["learning_rate"](i * config["epoque_iters"]),
            ", Mean accuracy (%):",
            (1 - jnp.mean(mean_abs_loss)) * 100,
        )

        if i % config["reset_env_nepoques"] == 0:
            env_states = jnp.zeros_like(env_states)
            print("ENV RESET")

    # Print best result
    print("Maximum accuracy attained in training:", max(mean_accuracy))

    # Checkpoint
    checkpoints.save_checkpoint(
        ckpt_dir=config["working_dir"] + config["run_name"],
        target=params,
        step=config["n_epoques"] * config["epoque_iters"],
    )

    # Plots
    plt.plot([i for i in range(len(mean_losses[100:]))], mean_losses[100:])
    plt.xlabel("Steps")
    plt.ylabel("Mean Losses")
    plt.savefig(config["working_dir"] + config["run_name"] + "/mean_losses.jpg")
    plt.close()

    plt.plot([i for i in range(len(mean_accuracy[100:]))], mean_accuracy[100:])
    plt.xlabel("Steps")
    plt.ylabel("Mean Accuracy")
    plt.savefig(config["working_dir"] + config["run_name"] + "/mean_accuracy.jpg")
    plt.close()

    return params, optim, nn_policy, mean_losses, mean_accuracy


def main():
    """Confg dictionary"""

    # learning rate schedule (just put the number if you want it fixed)
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(0.0001),
            optax.constant_schedule(0.00001),
            optax.constant_schedule(0.000001),
            optax.constant_schedule(0.0000008),
            optax.constant_schedule(0.0000004),
        ],
        boundaries=[200000, 400000, 600000, 800000],
    )

    # Now we create a config dict
    config = {
        "n_batches": 1,  # number of minibatches per device ( for a total of 8*n_batches batches)
        "batch_size": 64,  # size of each minibatch
        "layers": [1024, 1024],  # layers of the NN
        "epoque_iters": 1000,  # frequency at which we print mean loss
        "n_epoques": 1000,  # number of log cycles (4000)
        # (if epoque_iters =100, and n_epoques=1000, total iters are 100000)
        "learning_rate": lr_schedule,
        "seed": 260,  # random seed, set to whatever int.
        "reset_env_nepoques": 10000,
        "run_name": "run_correct_1",
        "date": "August_14",
        "working_dir": "/content/drive/MyDrive/Jaxecon/Pretraining/",
    }

    # Print some key statistics
    print(
        "Number of parameters of NN:",
        jnp.sum(
            jnp.array(
                [
                    (config["layers"][i] + 1) * config["layers"][i + 1]
                    for i in range(len(config["layers"]) - 1)
                ]
            )
        ),
    )
    cores_count = len(jax.devices())

    num_steps_perepisode = cores_count * config["batch_size"] * config["n_batches"]
    print("periods per episode:", num_steps_perepisode)

    num_steps_percycle = (
        cores_count
        * config["epoque_iters"]
        * config["batch_size"]
        * config["n_batches"]
    )
    print("periods per epoque:", num_steps_percycle)

    """ Run experiemnt """
    params, optim, nn_policy, mean_losses, mean_accuracy = run_experiment(
        ProdNetRbc_pretrain(), config
    )


if __name__ == "__main__":
    main()
