import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import torch
from datasets import load_dataset
from einops import rearrange, reduce
from torch.utils.data import DataLoader
from tqdm import tqdm

from x_mlps import XMLP, create_shift1d_op, layernorm_factory, s2mlp_block_factory

# Training parameters
BATCH_SIZE = 256
NUM_EPOCHS = 30
MIN_SNR = 0.0
# Model parameters
PATCH_SIZE = 16
DIM = 256
DEPTH = 8
# Optimizer parameters
INIT_VALUE = 0
PEAK_VALUE = 3e-3
WARMUP_EPOCHS = 5
CLIPPING = 0.16


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

    return hk.vmap(model_fn, in_axes=(0, None))


def create_loss_fn(num_classes: int = 24, alpha: float = 0.1, reduction: str = "mean"):
    @jax.jit
    def loss_fn(loss_scale: jmp.LossScale, y_hat: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        y = jax.nn.one_hot(y, num_classes)
        y = optax.smooth_labels(y, alpha=alpha)

        loss = optax.softmax_cross_entropy(y_hat, y)
        loss = reduce(loss, "... c -> ...", reduction)
        return loss_scale.scale(loss)

    return loss_fn


def create_step_fn(loss_fn, optimizer: optax.GradientTransformation):
    @jax.jit
    def step_fn(
        params: hk.Params,
        rng: jax.random.KeyArray,
        opt_state: optax.OptState,
        loss_scale: jmp.LossScale,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, rng, loss_scale, x, y)
        loss_value, grads = loss_scale.unscale(loss_value), loss_scale.unscale(grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        grads_finite = jmp.all_finite(grads)
        loss_scale = loss_scale.adjust(grads_finite)
        new_params, new_opt_state = jmp.select_tree(
            grads_finite,
            (new_params, new_opt_state),
            (params, opt_state),
        )

        return (new_params, new_opt_state, loss_scale), loss_value

    return step_fn


@jax.jit
def normalize(x: jnp.ndarray) -> jnp.ndarray:
    power = jnp.mean(jnp.abs(x) ** 2, axis=1, keepdims=True)
    x = x / jnp.sqrt(power)
    return jnp.stack([x.real, x.imag], axis=-1)


def fit(
    model_fn,
    loss_fn,
    optimizer: optax.GradientTransformation,
    params: hk.Params,
    opt_state: optax.OptState,
    loss_scale: jmp.LossScale,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rng: jax.random.KeyArray,
    num_epochs: int = 1,
):
    def forward(
        params: hk.Params, rng: jax.random.KeyArray, loss_scale: jmp.LossScale, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        y_hat = model_fn(params, rng, x, True)
        return loss_fn(loss_scale, y_hat, y)

    @jax.jit
    def predict(params: hk.Params, x: jnp.ndarray) -> jnp.ndarray:
        return model_fn(params, None, x, False)

    step = create_step_fn(forward, optimizer)

    # Ensure the model can even be run.
    for i, (x, _) in enumerate(val_loader):
        x = jnp.array(x)
        x = normalize(x)
        _ = predict(params, x)
        if i >= 2:
            break

    for epoch in range(num_epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as t:
            for x, y in train_loader:
                x, y = jnp.array(x), jnp.array(y)
                x = normalize(x)
                rng, subkey = jax.random.split(rng)
                (params, opt_state, loss_scale), loss_value = step(params, subkey, opt_state, loss_scale, x, y)
                t.set_postfix(loss=loss_value)
                t.update()
            num_correct, num_samples = 0, 0
            losses = []
            for x, y in val_loader:
                x, y = jnp.array(x), jnp.array(y)
                x = normalize(x)
                y_hat = predict(params, x)
                loss_value = loss_fn(loss_scale, y_hat, y)

                losses.append(loss_scale.unscale(loss_value).item())
                num_correct += jnp.sum(jnp.argmax(y_hat, axis=-1) == y).item()
                num_samples += x.shape[0]
            accuracy = num_correct / num_samples
            t.set_postfix(val_loss=np.mean(losses), val_acc=accuracy)

    return params, opt_state


def main():
    torch.manual_seed(0)
    # Load datasets
    train = load_dataset("data/train.h5", min_snr=MIN_SNR)
    val = load_dataset("data/valid.h5", min_snr=MIN_SNR, in_memory=True)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, collate_fn=collate_fn)
    # Load AMP policy
    amp_policy = jmp.get_policy("p=f32,c=f16,o=f32")
    hk.mixed_precision.set_policy(XMLP, amp_policy)
    hk.mixed_precision.set_policy(GELU, jmp.get_policy("f32"))
    loss_scale = jmp.DynamicLossScale(jmp.half_dtype()(2 ** 15))

    # Create and initalize model
    model = create_model(patch_size=PATCH_SIZE, dim=DIM, depth=DEPTH)
    model_fn = hk.transform(model)

    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    params = model_fn.init(subkey, jnp.ones((1, 1024, 2)), False)

    # Create and initialize optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=INIT_VALUE,
        peak_value=PEAK_VALUE,
        warmup_steps=len(train_loader) * WARMUP_EPOCHS,
        decay_steps=len(train_loader) * NUM_EPOCHS,
    )
    optimizer = optax.chain(optax.adaptive_grad_clip(clipping=CLIPPING), optax.lamb(schedule))

    opt_state = optimizer.init(params)

    # Train!
    loss_fn = create_loss_fn()
    params, opt_state = fit(
        model_fn.apply,
        loss_fn,
        optimizer,
        params,
        opt_state,
        loss_scale,
        train_loader,
        val_loader,
        rng=key,
        num_epochs=NUM_EPOCHS,
    )


if __name__ == "__main__":
    main()
