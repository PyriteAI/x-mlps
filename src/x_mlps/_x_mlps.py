from typing import Any, Callable, Literal, Optional, Protocol, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
from einops import reduce

from ._utils import group_by_prefix_and_trim

FeedForwardFactory = Callable[[int, int, int], hk.Module]
NormFactory = Callable[[int, int, int], hk.Module]
SublayerFactory = Callable[[int, int, int], hk.Module]
FFStrategy = Literal["mlpmixer", "resmlp"]


class BlockFactory(Protocol):
    def __call__(self, num_patches: int, dim: int, depth: int, **kwargs: Any) -> hk.Module:
        ...


def _calc_layer_scale_eps(depth: int) -> float:
    if depth <= 18:
        init_eps = 0.1
    elif depth > 18 and depth <= 24:
        init_eps = 1e-5
    else:
        init_eps = 1e-6
    return init_eps


class Affine(hk.Module):
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
    def __init__(self, dim: int, depth: int, name: Optional[str] = None):
        super().__init__(name=name)

        self.dim = dim
        self.init_eps = _calc_layer_scale_eps(depth)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = (1,) * (inputs.ndim - 1) + (self.dim,)
        s = hk.get_parameter("s", shape=shape, init=hk.initializers.Constant(self.init_eps))

        return s * inputs


class MLPMixerXPatchFeedForward(hk.Module):
    def __init__(
        self,
        dim: int,
        dim_hidden: Optional[int] = None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)

        self.dim = dim
        self.dim_hidden = dim * 4 if dim_hidden is None else dim_hidden
        self.activation = activation

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = hk.Conv1D(self.dim_hidden, 1, data_format="NCW", name="conv_1")(inputs)
        x = self.activation(x)
        return hk.Conv1D(self.dim, 1, data_format="NCW", name="conv_2")(x)


class ResMLPXPatchFeedForward(hk.Module):
    def __init__(self, dim: int, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)

        self.dim = dim

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return hk.Conv1D(self.dim, 1, data_format="NCW", name="conv")(inputs)


class XChannelFeedForward(hk.Module):
    def __init__(
        self,
        dim: int,
        dim_hidden: Optional[int] = None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.dim = dim
        self.dim_hidden = dim * 4 if dim_hidden is None else dim_hidden
        self.activation = activation

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.dim_hidden, name="linear_1")(inputs)
        x = self.activation(x)
        x = hk.Linear(self.dim, name="linear_2")(x)
        return x


class XSublayer(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        feedforward: FeedForwardFactory,
        prenorm: Optional[NormFactory] = None,
        postnorm: Optional[NormFactory] = None,
        residual: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.feedforward = feedforward
        self.prenorm = prenorm
        self.postnorm = postnorm
        self.residual = residual

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        if self.prenorm is not None:
            x = self.prenorm(self.num_patches, self.dim, self.depth)(x)
        x = self.feedforward(self.num_patches, self.dim, self.depth)(x)
        if self.postnorm is not None:
            x = self.postnorm(self.num_patches, self.dim, self.depth)(x)
        if self.residual:
            x += inputs

        return x


class XBlock(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        sublayers: Sequence[SublayerFactory],
        residual: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.sublayers = tuple(sublayers)
        self.residual = residual

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for Sublayer in self.sublayers:
            x = Sublayer(self.num_patches, self.dim, self.depth)(x)
        if self.residual:
            x += inputs
        return x


class XMLP(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        block: BlockFactory,
        normalization: Optional[NormFactory] = None,
        num_classes: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.block = block
        self.normalization = normalization
        self.num_classes = num_classes
        self.block_kwargs = kwargs

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.dim, name="proj_in")(inputs)
        for i in range(self.depth):
            x = self.block(self.num_patches, self.dim, i + 1, **self.block_kwargs)(x)
        if self.normalization is not None:
            x = self.normalization(self.num_patches, self.dim, self.depth + 1)(x)
        if self.num_classes is not None:
            x = reduce(x, "... n c -> ... c", "mean")
            x = hk.Linear(self.num_classes, name="proj_out")(x)
        return x


def mlpmixer_block_factory(num_patches: int, dim: int, depth: int, **kwargs: Any) -> hk.Module:
    xpatch_kwargs, kwargs = group_by_prefix_and_trim("xpatch_", kwargs)
    xpatch_ff_kwargs, xpatch_kwargs = group_by_prefix_and_trim("ff_", xpatch_kwargs)

    return XBlock(
        num_patches,
        dim,
        depth,
        [
            # Cross patch sublayer
            lambda num_patches, dim, depth: XSublayer(
                num_patches,
                dim,
                depth,
                feedforward=lambda num_patches, dim, depth: MLPMixerXPatchFeedForward(
                    num_patches, **xpatch_ff_kwargs, name="mlpmixer_xpatch_ff"
                ),
                prenorm=lambda *_: hk.LayerNorm(-1, create_scale=True, create_offset=True, name="mlpmixer_xpatch_ln"),
                name=f"mlpmixer_xpatch_{depth}",
            ),
            # Cross channel sublayer
            lambda num_patches, dim, depth: XSublayer(
                num_patches,
                dim,
                depth,
                feedforward=lambda num_patches, dim, depth: XChannelFeedForward(dim, name="xchannel_ff"),
                prenorm=lambda *_: hk.LayerNorm(-1, create_scale=True, create_offset=True, name="xchannel_ln"),
                name=f"xchannel_{depth}",
            ),
        ],
        name=f"mlpmixer_block_{depth}",
    )


def resmlp_block_factory(num_patches: int, dim: int, depth: int, **kwargs: Any) -> hk.Module:
    return XBlock(
        num_patches,
        dim,
        depth,
        [
            lambda num_patches, dim, depth: XSublayer(
                num_patches,
                dim,
                depth,
                feedforward=lambda num_patches, dim, depth: ResMLPXPatchFeedForward(
                    num_patches, name="resmlp_xpatch_ff"
                ),
                prenorm=lambda num_patches, dim, depth: Affine(dim, name="resmlp_xpatch_affine"),
                postnorm=lambda num_patches, dim, depth: LayerScale(dim, depth, name="resmlp_xpatch_scale"),
                name=f"resmlp_xpatch_{depth}",
            ),
            lambda num_patches, dim, depth: XSublayer(
                num_patches,
                dim,
                depth,
                feedforward=lambda num_patches, dim, depth: XChannelFeedForward(dim, name="xchannel_ff"),
                prenorm=lambda num_patches, dim, depth: Affine(dim, name="xchannel_affine"),
                postnorm=lambda num_patches, dim, depth: LayerScale(dim, depth, name="xchannel_scale"),
                name=f"xchannel_{depth}",
            ),
        ],
        name=f"resmlp_block_{depth}",
    )


__all__ = [
    "Affine",
    "LayerScale",
    "MLPMixerXPatchFeedForward",
    "ResMLPXPatchFeedForward",
    "XBlock",
    "XChannelFeedForward",
    "XMLP",
    "XSublayer",
    "mlpmixer_block_factory",
    "resmlp_block_factory",
]
