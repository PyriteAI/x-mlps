import os
from collections import namedtuple
from functools import partial
from typing import Callable, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from einops import rearrange, reduce
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from x_mlps import XMLP, create_shift2d_op, layernorm_factory, s2mlp_block_factory

StepState = namedtuple("StepState", ["params", "opt_state"])


def _get_interpolation_mode(interpolation: str):
    if interpolation == "bicubic":
        mode = transforms.InterpolationMode.BICUBIC
    elif interpolation == "bilinear":
        mode = transforms.InterpolationMode.BILINEAR
    elif interpolation == "nearest":
        mode = transforms.InterpolationMode.NEAREST
    else:
        raise ValueError(f"Unknown interpolation mode: {interpolation}")
    return mode


def create_convnext_train_transform(
    input_size: int = 224,
    color_jitter: float = 0.4,
    random_augment: str = "rand-m9-mstd0.5-inc1",
    interpolation: str = "bicubic",
    re_prob: float = 0.25,
    re_mode: str = "pixel",
    re_count: int = 1,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    return create_transform(
        input_size=input_size,
        color_jitter=color_jitter,
        auto_augment=random_augment,
        interpolation=interpolation,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        mean=mean,
        std=std,
        is_training=True,
    )


def create_eval_transform(
    input_size: int = 224,
    crop_pct: float = 0.875,
    interpolation: str = "bicubic",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    eval_transforms = []
    if input_size >= 384:
        eval_transforms.append(
            transforms.Resize((input_size, input_size), interpolation=_get_interpolation_mode(interpolation.lower()))
        )
    else:
        size = int(input_size / crop_pct)
        eval_transforms.append(transforms.Resize(size, interpolation=_get_interpolation_mode(interpolation.lower())))
        eval_transforms.append(transforms.CenterCrop(input_size))
    eval_transforms.append(transforms.ToTensor())
    eval_transforms.append(transforms.Normalize(mean, std))
    return transforms.Compose(eval_transforms)


def create_model(
    patch_size: int, dim: int, depth: int, num_classes: int = 1000, stochastic_depth: Union[float, bool] = 0.5
):
    def model_fn(x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        x = rearrange(x, "... (h p1) (w p2) c -> ... h w (p1 p2 c)", p1=patch_size, p2=patch_size)
        x = XMLP(
            num_patches=x.shape[-2],
            dim=dim,
            depth=depth,
            block=s2mlp_block_factory,
            normalization=layernorm_factory,
            stochastic_depth=0.5,
            block_drop_path_mode="sample",
            block_sublayer1_ff_shift=create_shift2d_op(),
        )(x, is_training=is_training)
        x = x = reduce(x, "... h w c -> ... c", "mean")
        return hk.Linear(num_classes, name="proj_out")(x)

    return model_fn


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
        params = model.init(weight_rng, jnp.ones((1, 224, 224, 3)), False)

        # Create and initialize optimizer
        optimizer = optimizer_factory()
        opt_state = optimizer.init(params)

        return params, opt_state

    return initialize


def create_loss_fn(reduction: str = "mean"):
    @jax.jit
    def loss_fn(y_hat: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
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
    num_classes: int = 1000,
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
        x = rearrange(x, "b c h w -> b h w c")
        x = jnp.array(x)
        _ = predict(eval_params, x)
        if i >= 2:
            break
    del eval_params

    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=num_classes)
    step_state = StepState(params, opt_state)
    with tqdm(total=len(train_loader) + len(val_loader), unit="batch") as t:
        for epoch in range(num_epochs):
            t.reset()
            t.set_description(f"Epoch {epoch + 1}")

            for x, y in train_loader:
                x, y = mixup(x, y)

                x = rearrange(x, "b c h w -> b h w c")
                x = jnp.array(x.view(local_device_count, -1, *x.shape[1:]))
                y = jnp.array(y.view(local_device_count, -1, *y.shape[1:]))
                rng, *subkeys = jax.random.split(rng, num=local_device_count + 1)
                subkeys = jnp.stack(subkeys, axis=0)  # very important - pmap doesn't like lists.
                step_state, loss_value = step(step_state, subkeys, x, y)
                t.set_postfix(loss=loss_value[0])
                t.update()
            eval_params = jax.tree_map(lambda x: x[0], step_state.params)
            num_correct, num_samples = 0, 0
            losses = []
            for x, y in val_loader:
                x = rearrange(x, "b c h w -> b h w c")
                x, y = jnp.array(x), jnp.array(y)

                y_hat = predict(eval_params, x)
                loss_value = loss_fn(y_hat, y)

                losses.append(loss_value.item())
                num_correct += jnp.sum(jnp.argmax(y_hat, axis=-1) == y).item()
                num_samples += x.shape[0]
                t.update()
            del eval_params
            accuracy = num_correct / num_samples
            t.set_postfix(val_loss=np.mean(losses), val_acc=accuracy)

    return params, opt_state


def main():
    torch.manual_seed(0)
    # Load datasets
    train_transform = create_convnext_train_transform()
    eval_transform = create_eval_transform()
    train = ImageFolder("data/imagenet1k/train", transform=train_transform)
    val = ImageFolder("data/imagenet1k/val", transform=eval_transform)
    train_loader = DataLoader(train, batch_size=256, shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=8)

    local_device_count = jax.local_device_count()

    # Create and initalize model.
    model = create_model(patch_size=14, dim=512, depth=8)
    model_fn = hk.transform(model)

    rng_key, weight_key = jax.random.split(jax.random.PRNGKey(0))
    # For initialization we need the same random key on each device.
    weight_key = jnp.broadcast_to(weight_key, (local_device_count,) + weight_key.shape)

    optimizer_factory = create_optimizer_factory_fn(0, 1e-3, len(train_loader) * 20, len(train_loader) * 300, 0.64)
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
        num_epochs=300,
    )


if __name__ == "__main__":
    main()
