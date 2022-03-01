from collections import namedtuple
from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from datasets import load_dataset
from einops import rearrange, reduce
from torch.utils.data import DataLoader
from tqdm import tqdm

from x_mlps import XMLP, create_shift1d_op, layernorm_factory, s2mlp_block_factory

# Training parameters
BATCH_SIZE = 4096
NUM_EPOCHS = 20
MIN_SNR = None
# Model parameters
PATCH_SIZE = 16
DIM = 256
DEPTH = 8
# Optimizer parameters
INIT_VALUE = 0
PEAK_VALUE = 8e-3
WARMUP_EPOCHS = 10
CLIPPING = 0.01


StepState = namedtuple("StepState", ["params", "opt_state"])


class GELU(hk.Module):
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.gelu(inputs)


def collate_fn(batch):
    data, target, _ = zip(*batch)

    return np.stack(data, axis=0), np.array(target)


def create_model(patch_size: int, dim: int, depth: int, num_classes: int = 24):
    def model_fn(x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        x = rearrange(x, "(n p) c -> n (p c)", p=patch_size)
        return XMLP(
            num_patches=x.shape[-2],
            dim=dim,
            depth=depth,
            block=s2mlp_block_factory,
            normalization=layernorm_factory,
            num_classes=num_classes,
            block_sublayers_ff_activation=GELU(),
            block_sublayer1_ff_shift=create_shift1d_op(),
        )(x, is_training=is_training)

    return hk.vmap(model_fn, in_axes=(0, None), axis_name="local_batch")


def create_optimizer_factory_fn(
    init_value: float, peak_value: float, warmup_steps: int, decay_steps: int, clipping: float
):
    def create_optimizer() -> optax.GradientTransformation:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=init_value, peak_value=peak_value, warmup_steps=warmup_steps, decay_steps=decay_steps
        )
        optimizer = optax.chain(optax.adaptive_grad_clip(clipping=clipping), optax.lamb(schedule))

        return optimizer

    return create_optimizer


def create_initialization_fn(model: hk.Transformed, optimizer_factory: Callable[[], optax.GradientTransformation]):
    @partial(jax.pmap, axis_name="i")
    def initialize(weight_rng: jnp.ndarray) -> Tuple[hk.Params, optax.OptState]:
        params = model.init(weight_rng, jnp.ones((1, 1024, 2)), False)

        # Create and initialize optimizer
        optimizer = optimizer_factory()
        opt_state = optimizer.init(params)

        return params, opt_state

    return initialize


def create_loss_fn(num_classes: int = 24, alpha: float = 0.1, reduction: str = "mean"):
    @jax.jit
    def loss_fn(y_hat: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        y = jax.nn.one_hot(y, num_classes)
        y = optax.smooth_labels(y, alpha=alpha)

        loss = optax.softmax_cross_entropy(y_hat, y)
        return reduce(loss, "... c -> ...", reduction)

    return loss_fn


def create_step_fn(loss_fn, optimizer_factory: Callable[[], optax.GradientTransformation]):
    @partial(jax.pmap, axis_name="i")
    def step_fn(state: StepState, rng: jax.random.KeyArray, x: jnp.ndarray, y: jnp.ndarray):
        loss_value, grads = jax.value_and_grad(loss_fn)(state.params, rng, x, y)
        loss_value = jax.lax.pmean(loss_value, axis_name="i")
        grads = jax.lax.pmean(grads, axis_name="i")

        updates, new_opt_state = optimizer_factory().update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        return StepState(new_params, new_opt_state), loss_value

    return step_fn


@jax.jit
def normalize(x: jnp.ndarray) -> jnp.ndarray:
    power = jnp.mean(jnp.abs(x) ** 2, axis=1, keepdims=True)
    x = x / jnp.sqrt(power)
    return jnp.stack([x.real, x.imag], axis=-1)


def fit(
    model_fn,
    loss_fn,
    optimizer_factory: Callable[[], optax.GradientTransformation],
    params: hk.Params,
    opt_state: optax.OptState,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rng: jax.random.KeyArray,
    num_epochs: int = 1,
):
    def forward(params: hk.Params, rng: jax.random.KeyArray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        y_hat = model_fn(params, rng, x, True)
        return loss_fn(y_hat, y)

    @jax.jit
    def predict(params: hk.Params, x: jnp.ndarray) -> jnp.ndarray:
        return model_fn(params, None, x, False)

    step = create_step_fn(forward, optimizer_factory)

    local_device_count = jax.local_device_count()
    # Ensure the model can even be run.
    eval_params = jax.tree_map(lambda x: x[0], params)
    for i, (x, _) in enumerate(val_loader):
        x = normalize(x)
        _ = predict(eval_params, x)
        if i >= 2:
            break
    del eval_params

    step_state = StepState(params, opt_state)
    for epoch in range(num_epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as t:
            for x, y in train_loader:
                x = normalize(x)
                x = x.reshape(local_device_count, -1, *x.shape[1:])
                y = y.reshape(local_device_count, -1)
                rng, *subkeys = jax.random.split(rng, num=local_device_count + 1)
                subkeys = jnp.stack(subkeys, axis=0)  # very important - pmap doesn't like lists.
                step_state, loss_value = step(step_state, subkeys, x, y)
                t.set_postfix(loss=loss_value[0])
                t.update()
            eval_params = jax.tree_map(lambda x: x[0], step_state.params)
            num_correct, num_samples = 0, 0
            losses = []
            for x, y in val_loader:
                x = normalize(x)
                y_hat = predict(eval_params, x)
                loss_value = loss_fn(y_hat, y)

                losses.append(loss_value.item())
                num_correct += jnp.sum(jnp.argmax(y_hat, axis=-1) == y).item()
                num_samples += x.shape[0]
            del eval_params
            accuracy = num_correct / num_samples
            t.set_postfix(val_loss=np.mean(losses), val_acc=accuracy)

    return params, opt_state


def main():
    torch.manual_seed(0)
    # Load datasets
    train = load_dataset("data/train.h5", min_snr=MIN_SNR, in_memory=True)
    val = load_dataset("data/valid.h5", min_snr=MIN_SNR, in_memory=True)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, collate_fn=collate_fn)

    local_device_count = jax.local_device_count()

    # Create and initalize model.
    model = create_model(patch_size=PATCH_SIZE, dim=DIM, depth=DEPTH)
    model_fn = hk.transform(model)

    rng_key, weight_key = jax.random.split(jax.random.PRNGKey(0))
    # For initialization we need the same random key on each device.
    weight_key = jnp.broadcast_to(weight_key, (local_device_count,) + weight_key.shape)

    optimizer_factory = create_optimizer_factory_fn(
        INIT_VALUE, PEAK_VALUE, len(train_loader) * WARMUP_EPOCHS, len(train_loader) * NUM_EPOCHS, CLIPPING
    )
    params, opt_state = create_initialization_fn(model_fn, optimizer_factory)(weight_key)

    num_params = sum(x.size for x in jax.tree_leaves(params))
    print(f"Parameter count: {num_params}")

    # Train!
    loss_fn = create_loss_fn()
    params, opt_state = fit(
        model_fn.apply,
        loss_fn,
        optimizer_factory,
        params,
        opt_state,
        train_loader,
        val_loader,
        rng=rng_key,
        num_epochs=NUM_EPOCHS,
    )


if __name__ == "__main__":
    main()
