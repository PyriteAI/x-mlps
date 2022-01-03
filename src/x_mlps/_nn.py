from typing import Any, Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange

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


def create_shift2d_op(height: int, width: int, amount: int = 1) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create a 2D shift operator based on spatial shift algorithm introduced in S^2-MLP¹.

    Args:
        height: Height of the original input image divided by the patch size.
        width: Width of the original input image divided by the patch size.
        amount: Amount of shift.

    Returns:
        Callable[[jnp.ndarray], jnp.ndarray]: The configured shift operator.

    References:
        1. S2-MLP: Spatial-Shift MLP Architecture for Vision (https://arxiv.org/abs/2106.07477).
    """

    def shift2d(x: jnp.ndarray) -> jnp.ndarray:
        x = rearrange(x, "... (h w) c -> ... h w c", h=height, w=width)
        x1, x2, x3, x4 = jnp.split(x, 4, axis=-1)
        x1 = x1.at[amount:].set(x1[:-amount])
        x2 = x2.at[:-amount].set(x2[amount:])
        x3 = x3.at[:, amount:].set(x3[:, :-amount])
        x4 = x4.at[:, :-amount].set(x4[:, amount:])
        x = jnp.concatenate([x1, x2, x3, x4], axis=-1)
        x = rearrange(x, "... h w c -> ... (h w) c")
        return x

    return shift2d


class SampleDropout(hk.Module):
    """Randomly drop the input with a given probability.

    This is equivalent to Stochastic Depth when applied to the output of a network path¹.

    Args:
        rate (float): Probability of dropping an element.
        name (str, optional): Name of the module.

    References:
        1. Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382).
    """

    def __init__(self, rate: float, name: Optional[str] = None):
        super().__init__(name=name)

        self.rate = rate

    def __call__(self, x: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        if is_training:
            return hk.cond(
                jax.random.bernoulli(hk.next_rng_key(), 1 - self.rate), lambda x: x, lambda x: jnp.zeros_like(x), x
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
    "SampleDropout",
    "SpatialGatingUnit",
    "layernorm_factory",
    "create_shift2d_op",
    "sgu_factory",
]
