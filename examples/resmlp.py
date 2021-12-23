import os

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from einops import rearrange, reduce
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from x_mlps import XMLP

# Model parameters
PATCH_SIZE = 4
DIM = 384
DEPTH = 12
# Optimizer parameters
INIT_VALUE = 0
PEAK_VALUE = 3e-4
WARMUP_STEPS = 2000
CLIPPING = 0.32
# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 100


def collate_fn(batch):
    data, target = zip(*batch)

    return np.stack(data, axis=0), np.array(target)


def create_model(patch_size: int, dim: int, depth: int, num_classes: int = 10):
    @hk.vmap
    def model_fn(x: jnp.ndarray) -> jnp.ndarray:
        x = rearrange(x, "(h p1) (w p2) c -> (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
        return XMLP(num_patches=x.shape[-2], dim=dim, depth=depth, num_classes=num_classes)(x)

    return model_fn


def create_loss_fn(num_classes: int = 10, alpha: float = 0.1, reduction: str = "mean"):
    @jax.jit
    def loss_fn(y_hat: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        y = jax.nn.one_hot(y, num_classes)
        y = optax.smooth_labels(y, alpha=alpha)

        loss = optax.softmax_cross_entropy(y_hat, y)
        return reduce(loss, "... c -> ...", reduction)

    return loss_fn


def create_step_fn(loss_fn, optimizer: optax.GradientTransformation):
    @jax.jit
    def step_fn(params: hk.Params, opt_state: optax.OptState, x: jnp.ndarray, y: jnp.ndarray):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_value

    return step_fn


def fit(
    model_fn,
    loss_fn,
    optimizer: optax.GradientTransformation,
    params: hk.Params,
    opt_state: optax.OptState,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 1,
):
    def forward(params: hk.Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        y_hat = model_fn(params, x)
        return loss_fn(y_hat, y)

    step = create_step_fn(forward, optimizer)

    for epoch in range(num_epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as t:
            for x, y in train_loader:
                x, y = jnp.array(x), jnp.array(y)
                params, opt_state, loss_value = step(params, opt_state, x, y)
                t.set_postfix(loss=loss_value)
                t.update()
            num_correct, num_samples = 0, 0
            losses = []
            for x, y in val_loader:
                x, y = jnp.array(x), jnp.array(y)
                y_hat = model_fn(params, x)
                loss_value = loss_fn(y_hat, y)

                losses.append(loss_value.item())
                num_correct += jnp.sum(jnp.argmax(y_hat, axis=-1) == y).item()
                num_samples += x.shape[0]
            accuracy = num_correct / num_samples
            t.set_postfix(val_loss=np.mean(losses), val_acc=accuracy)

    return params, opt_state


def main():
    torch.manual_seed(0)
    # Load datasets
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cifar10")
    train = CIFAR10(
        root,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda img: np.array(img)),
            ]
        ),
        download=True,
    )
    val = CIFAR10(root, train=False, transform=transforms.Lambda(lambda img: np.array(img)), download=True)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # Create and initalize model
    model = create_model(patch_size=PATCH_SIZE, dim=DIM, depth=DEPTH)
    model_fn = hk.transform(model)
    model_fn = hk.without_apply_rng(model_fn)

    rng = jax.random.PRNGKey(0)
    params = model_fn.init(rng, jnp.ones((1, 32, 32, 3)))
    # Create and initialize optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=INIT_VALUE,
        peak_value=PEAK_VALUE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=len(train_loader) * NUM_EPOCHS,
    )
    optimizer = optax.chain(optax.adaptive_grad_clip(clipping=CLIPPING), optax.adamw(schedule))

    opt_state = optimizer.init(params)

    # Train!
    loss_fn = create_loss_fn()
    params, opt_state = fit(
        model_fn.apply, loss_fn, optimizer, params, opt_state, train_loader, val_loader, num_epochs=NUM_EPOCHS
    )


if __name__ == "__main__":
    main()
