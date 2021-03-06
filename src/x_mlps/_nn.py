from typing import Any, Callable, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

from ._types import XModuleFactory
from ._utils import pick_and_pop


def _calc_layer_scale_eps(depth: int) -> float:
    if depth <= 18:
        init_eps = 0.1
    elif depth > 18 and depth <= 24:
        init_eps = 1e-5
    else:
        init_eps = 1e-6
    return init_eps


def create_multishift1d_op(amount: Sequence[int], bidirectional: bool = True) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create a 1D multi-shift operator, inspired by the spatial shift operator introduced in S^2-MLP¹.

    This operation works by calling `create_shift1d_op` with each value in `amount` and then concatenating the results.
    By default the shift operator is bidirectional, resulting in interleaved shifting.

    Expects inputs to be of shape `(num_patches, dim)` or `(batch_size, num_patches, dim)`.

    Args:
        amount: Sequence of shift amounts.
        bidirectional: Whether to use bidirectional shift. Defaults to True.

    Returns:
        The configured shift operator.

    References:
        1. S2-MLP: Spatial-Shift MLP Architecture for Vision (https://arxiv.org/abs/2106.07477).
    """

    def multishift1d(x: jnp.ndarray) -> jnp.ndarray:
        xs = jnp.split(x, len(amount), axis=-1)
        shifted_xs = [create_shift1d_op(amount[i], bidirectional=bidirectional)(x) for i, x in enumerate(xs)]
        x = jnp.concatenate(shifted_xs, axis=-1)
        return x

    return multishift1d


def create_shift1d_op(amount: int = 1, bidirectional: bool = True) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create a 1D shift operator, inspired by the spatial shift operator introduced in S^2-MLP¹.

    By default the the shift operator is bidirectional. This results in the input data being split into two parts,
    one part is shifted forward and the other part is shifted backward by `amount`. The two parts are then
    concatenated. It's rare that unidirectional shift is needed; however, it can be useful as a building block for more
    complex shifting operations.

    Expects inputs to be of shape `(..., num_patches, dim)`.

    Args:
        amount: Amount of shift. Defaults to 1.
        bidirectional: Whether to use bidirectional shift. Defaults to True.

    Returns:
        The configured shift operator.

    References:
        1. S2-MLP: Spatial-Shift MLP Architecture for Vision (https://arxiv.org/abs/2106.07477).
    """

    def shift1d(x: jnp.ndarray) -> jnp.ndarray:
        slice_from1 = (slice(None, None, None),) * (x.ndim - 2) + (slice(amount, None, None),)
        slice_end1 = (slice(None, None, None),) * (x.ndim - 2) + (slice(None, -amount, None),)
        if bidirectional:
            x1, x2 = jnp.split(x, 2, axis=-1)
            x1 = x1.at[slice_from1].set(x1[slice_end1])
            x2 = x2.at[slice_end1].set(x2[slice_from1])
            x = jnp.concatenate([x1, x2], axis=-1)
        else:
            x = x.at[slice_from1].set(x[slice_end1])
        return x

    return shift1d


def create_shift2d_op(amount: int = 1) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create a 2D shift operator based on spatial shift algorithm introduced in S^2-MLP¹.

    Expects inputs to be of shape `(..., height, width, dim)`.

    Args:
        amount: Amount of shift.

    Returns:
        Callable[[jnp.ndarray], jnp.ndarray]: The configured shift operator.

    References:
        1. S2-MLP: Spatial-Shift MLP Architecture for Vision (https://arxiv.org/abs/2106.07477).
    """

    def shift2d(x: jnp.ndarray) -> jnp.ndarray:
        x1, x2, x3, x4 = jnp.split(x, 4, axis=-1)

        # Handle variable tensor shapes.
        hslice_from1 = (slice(None, None, None),) * (x.ndim - 3) + (slice(amount, None, None),)
        hslice_end1 = (slice(None, None, None),) * (x.ndim - 3) + (slice(None, -amount, None),)
        wslice_from1 = (slice(None, None, None),) * (x.ndim - 2) + (slice(amount, None, None),)
        wslice_end1 = (slice(None, None, None),) * (x.ndim - 2) + (slice(None, -amount, None),)

        x1 = x1.at[hslice_from1].set(x1[hslice_end1])
        x2 = x2.at[hslice_end1].set(x2[hslice_from1])
        x3 = x3.at[wslice_from1].set(x3[wslice_end1])
        x4 = x4.at[wslice_end1].set(x4[wslice_from1])

        x = jnp.concatenate([x1, x2, x3, x4], axis=-1)
        return x

    return shift2d


class DropPath(hk.Module):
    """Randomly drop the input with a given probability.

    This is equivalent to Stochastic Depth when applied to the output of a network path¹. Additionally, individual
    samples can be dropped independently when `mode` is set to `sample` (the default is to drop the entire batch),
    mimicking the implementation from PyTorch Image Models². However, sample mode is not supported with `vmap`.

    Args:
        rate: Probability of dropping an element.
        mode: Whether to operate on individual samples in a batch or the whole batch. Defaults to "batch". Must be set
            "batch" when `vmap` is used.
        name: Name of the module.

    References:
        1. Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382).
        2. PyTorch Image Models (https://github.com/rwightman/pytorch-image-models/blob/e4360e6125bb0bb4279785810c8eb33b40af3ebd/timm/models/layers/drop.py#L137).
    """

    def __init__(self, rate: float, mode: str = "batch", name: Optional[str] = None):
        super().__init__(name=name)

        if mode not in {"batch", "sample"}:
            raise ValueError(f"invalid mode: {mode}")

        self.rate = rate
        self.mode = mode

    def __call__(self, x: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        if is_training:
            if self.mode == "batch":
                x = (
                    jax.random.bernoulli(hk.next_rng_key(), 1 - self.rate)
                    * x
                    / jnp.clip(1 - self.rate + 1e-12, a_min=0, a_max=1)
                )
            elif self.mode == "sample":
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                x = (
                    jax.random.bernoulli(hk.next_rng_key(), 1 - self.rate, shape=shape)
                    * x
                    / jnp.clip(1 - self.rate + 1e-12, a_min=0, a_max=1)
                )
        return x


class Affine(hk.Module):
    """Affine transform layer as described in ResMLP¹.

    Briefly, this operator rescales and shifts its input by a learned weight and bias of size `dim`.

    Args:
        dim (int): Size of the channel dimension.
        init_scale (float): Initial weight value of alpha.
        name (str, optional): The name of the module. Defaults to None.

    References:
        1. ResMLP: Feedforward networks for image classification with data-efficient training
            (https://arxiv.org/abs/2105.03404).
    """

    def __init__(self, dim: int, init_scale: float = 1.0, name: Optional[str] = None):
        super().__init__(name=name)

        self.dim = dim
        self.init_scale = init_scale

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = (1,) * (inputs.ndim - 1) + (self.dim,)
        w = hk.get_parameter("w", shape=shape, init=hk.initializers.Constant(self.init_scale))
        b = hk.get_parameter("b", shape=shape, init=jnp.zeros)

        return w * inputs + b


class LayerScale(hk.Module):
    """LayerScale layer as described in *Going deeper with Image Transformers*¹.

    Briefly, rescales the input by a learned weight of size `dim`.

    Args:
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. This is used to determine the
            initial weight values. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.

    References:
        1. Going deeper with Image Transformers (https://arxiv.org/abs/2103.17239).
    """

    def __init__(self, dim: int, depth: int, name: Optional[str] = None):
        super().__init__(name=name)

        self.dim = dim
        self.init_eps = _calc_layer_scale_eps(depth)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = (1,) * (inputs.ndim - 1) + (self.dim,)
        s = hk.get_parameter("s", shape=shape, init=hk.initializers.Constant(self.init_eps))

        return s * inputs


class SpatialGatingUnit(hk.Module):
    """Spatial Gating Unit as described in *Pay Attention to MLPs*¹.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        norm (XModuleFactory, optional): Normalization layer factory function. Defaults to LayerNorm via
            `layernorm_factory`.
        activation (Callable[[jnp.ndarray], jnp.ndarray], optional): Activation function. Applied to the gate values
            spatial projection. Defaults to the identity function.
        init_eps (float): Initial weight of the spatial projection layer. Scaled by the number of patches. Defaults to
            1e-3.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Currently unused.

    References:
        1. Pay Attention to MLPs (https://arxiv.org/abs/2105.08050).
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        norm: XModuleFactory = None,
        activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        init_eps: float = 1e-3,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        if kwargs:
            raise KeyError(f"unknown keyword arguments: {list(kwargs.keys())}")

        if norm is None:
            norm = layernorm_factory
        if activation is None:
            activation = lambda x: x  # noqa: E731

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.norm = norm
        self.activation = activation
        self.init_eps = init_eps / num_patches

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        u, v = jnp.split(inputs, 2, axis=-1)
        v = self.norm(self.num_patches, self.dim, self.depth, name="norm")(v)
        v = hk.Conv1D(
            self.num_patches,
            1,
            w_init=hk.initializers.RandomUniform(-self.init_eps, self.init_eps),
            b_init=jnp.ones,
            data_format="NCW",
            name="conv",
        )(v)
        return u * self.activation(v)


def layernorm_factory(num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
    """Layer normalization module factory function.

    Standard `hk.LayerNorm` arguments can be passed via the `kwargs` dictionary. If no arguments are passed, the
    following defaults are used:

    1. `axis` is set to `-1` (the last dimension).
    2. `create_scale` is set to `True`.
    3. `create_offset` is set to `True`.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input. Unused.
        dim (int): Size of the channel dimension. Unused.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
            Unused.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: `hk.LayerNorm` arguments.

    Returns:
        hk.Module: `hk.LayerNorm` module.
    """
    axis, create_scale, create_offset = pick_and_pop(["axis", "create_scale", "create_offset"], kwargs)
    if axis is None:
        axis = -1
    if create_scale is None:
        create_scale = True
    if create_offset is None:
        create_offset = True

    return hk.LayerNorm(axis, create_scale, create_offset, **kwargs, name=name)


def sgu_factory(num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
    """`SpatialGatingUnit` module factory function.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional `SpatialGatingUnit` arguments.

    Returns:
        hk.Module: `SpatialGatingUnit` module.
    """

    return SpatialGatingUnit(num_patches, dim, depth, **kwargs, name=name)


__all__ = [
    "Affine",
    "LayerScale",
    "DropPath",
    "SpatialGatingUnit",
    "layernorm_factory",
    "create_multishift1d_op",
    "create_shift1d_op",
    "create_shift2d_op",
    "sgu_factory",
]
