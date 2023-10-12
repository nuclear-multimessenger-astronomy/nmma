from typing import Sequence, Callable
import functools
import time

import jax
import jax.numpy as jnp

from flax import linen as nn  # Linen API
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
from flax.training.train_state import TrainState

from ml_collections import ConfigDict

# from clu import metrics
import optax

"""Dataset creation"""

# TODO - can be improved with batching (with tensorflow or pytorch datasets?)

"""Configs"""
# TODO - remove to separate place?

def get_default_config():
    """
    Default hyperparameter settings for an MLP trained on SVD dimensionally reduced data
    """
    config = ConfigDict()

    # Basic set up
    config.name = "MLP"  # in case we have multiple architectures, can use the name to specify the architecture
    config.act_func = nn.relu
    config.layer_sizes = [64, 128, 64, 10] # 10 is the standard SVD output in NMMA
    # Optimizer. TODO gather the info in a dict to pass to the optimizer without inferring kwargs
    config.optimizer = optax.adam
    config.learning_rate = 1e-2 # initial learning rate if using scheduler to decay
    # Datasets and training info
    config.batch_size = 128 # unused for now, how to use?
    config.nb_epochs = 1000
    config.nb_report = 100
    # For the scheduler:
    config.nb_epochs_decay = int(round(config.nb_epochs / 10))
    # TODO - add learning rate scheduler
    config.learning_rate_fn = None
    # ^ optax learning rate scheduler


    # Custom scheduler (work in progress...)
    config.fixed_lr = False
    config.my_scheduler = ConfigDict() # to gather parameters
    # In case of false fixed learning rate, will adapt lr based on following params for custom scheduler
    config.my_scheduler.counter = 0
    # ^ count epochs during training loop, in order to only reduce lr after x amount of steps
    config.my_scheduler.threshold = 0.995
    # ^ if best loss has not recently improved by this fraction, then reduce learning rate
    config.my_scheduler.multiplier = 0.5
    # ^ reduce lr by this factor if loss is not improving sufficiently
    config.my_scheduler.patience = 10
    # ^ amount of epochs to wait in loss curve before adapting lr if loss goes up
    # config.my_scheduler.burnin = 20
    # # ^ amount of epochs to wait at start before any adaptation is done
    config.my_scheduler.history = 10
    # ^ amount of epochs to "look back" in order to determine whether loss is improving or not

    return config


"""Neural network architectures"""

class MLP(nn.Module):

    layer_sizes: Sequence[int] # sizes of the hidden layers of the neural network
    act_func: Callable # activation function applied on each hidden layer

    def setup(self):
        self.layers = [nn.Dense(n) for n in self.layer_sizes]

    # @functools.partial(jax.jit, static_argnums=(2, 3))
    @nn.compact
    def __call__(self, x):
        """_summary_

        Args:
            x (data): Input data of the neural network.
        """

        for i, layer in enumerate(self.layers):
            # Apply the linear part of the layer's operation
            x = layer(x)
            # If not the output layer, apply the given activation function
            if i != len(self.layer_sizes) - 1:
                x = self.act_func(x)

        return x


class NeuralNetwork(nn.Module):
    """A very basic initial neural network used for testing the basic functionalities of Flax.

    Returns:
        NeuralNetwork: The architecture of the neural network
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=24)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=24)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


"""Training"""

def my_lr_scheduler(config: ConfigDict, state, train_losses) -> None:
    """Custom learning rate scheduler

    Args:
        config (ConfigDict): Configuration dict for experiments.
        train_losses (list): Train losses recorded so far during a training loop.

    Returns:
        None
    """

    # If fixed lr, then no scheduler: return old state again
    if config.fixed_lr:
        return state

    # Defined custom scheduler here. Compare history of loss curve to previous best
    patience = config.my_scheduler.patience
    new_state = state
    if config.my_scheduler.counter >= patience:
        print(train_losses)
        print(train_losses[-patience // 2:])
        print(train_losses[-patience : -patience // 2])
        current_best = jnp.min(train_losses[-patience // 2:])
        previous_best = jnp.min(train_losses[-patience:-patience // 2])

        # If we did not improve the test loss sufficiently, going to adapt LR
        if current_best / previous_best >= config.my_scheduler.threshold:
            # Reset counter (note: will increment later, so set to -1 st it becomes 0)
            config.my_scheduler.counter = -1
            config.learning_rate = config.my_scheduler.multiplier * config.learning_rate
            # Reset optimizer
            tx = config.optimizer(learning_rate = config.learning_rate)
            new_state = TrainState.create(apply_fn = state.apply_fn, params = state.params, tx = tx)

    # Add to epoch counter for the scheduler
    config.my_scheduler.counter += 1
    return new_state



def create_train_state(model, test_input, rng, config):
    """
    Creates an initial `TrainState` from NN model and optimizer. Test input and RNG for initialization of the NN.
    TODO add Optax scheduler possibility here
    """
    # Initialize the parameters by passing dummy input
    params = model.init(rng, test_input)['params']
    tx = config.optimizer(config.learning_rate)
    state = TrainState.create(apply_fn = model.apply, params = params, tx = tx)
    return state

def apply_model(state, x_batched, y_batched):

    def loss_fn(params):
        def squared_error(x, y):
            # For a single datapoint
            pred = state.apply_fn({'params': params}, x)
            return jnp.inner(y - pred, y - pred) / 2.0
        # Vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return loss, grads

@jax.jit
def train_step(state, train_X, train_y, val_X = None, val_y = None):
    """
    Train for a single step. Note that this function is functionally pure and hence suitable for jit.
    """

    # Compute losses
    train_loss, grads = apply_model(state, train_X, train_y)
    if val_X is not None:
        val_loss, grads = apply_model(state, val_X, val_y)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    return state, train_loss, val_loss

# @jax.jit
def train_loop(state: TrainState, train_X, train_y, val_X = None, val_y = None, config = None):

    train_losses, val_losses = [], []

    if config is None:
        config = get_default_config()

    start = time.time()
    for i in range(config.nb_epochs):
        # Do a single step
        state, train_loss, val_loss = train_step(state, train_X, train_y, val_X, val_y)
        # Save the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # Report once in a while
        if i % config.nb_report == 0:
            print(f"Train loss at step {i+1}: {train_loss}")
            print(f"Valid loss at step {i+1}: {val_loss}")
            print(f"Learning rate: {config.learning_rate}")
            print("---")
        # TODO add my custom scheduler here?

    end = time.time()
    print(f"Training for {config.nb_epochs} took {end-start} seconds.")

    return state, train_losses, val_losses

