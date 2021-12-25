from typing import Any, Callable, Optional, Protocol, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange, reduce

from ._utils import group_by_prefix_and_trim, pick_and_pop


class XModuleFactory(Protocol):
    def __call__(self, num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
        ...


def _calc_layer_scale_eps(depth: int) -> float:
    if depth <= 18:
        init_eps = 0.1
    elif depth > 18 and depth <= 24:
        init_eps = 1e-5
    else:
        init_eps = 1e-6
    return init_eps


def create_shift2d_op(height: int, width: int, amount: int = 1) -> Callable:
    def shift2d(x: jnp.ndarray) -> jnp.ndarray:
        c = x.shape[-1]
        x = rearrange(x, "... (h w) c -> ... h w c", h=height, w=width)
        x = x.at[amount:, :, : c // 4].set(x[:-amount, :, : c // 4])
        x = x.at[:-amount, :, c // 4 : c // 2].set(x[amount:, :, c // 4 : c // 2])
        x = x.at[:, amount:, c // 2 : 3 * c // 4].set(x[:, :-amount, c // 2 : 3 * c // 4])
        x = x.at[:, :-amount, 3 * c // 4 : c].set(x[:, amount:, 3 * c // 4 : c])
        x = rearrange(x, "... h w c -> ... (h w) c")
        return x

    return shift2d


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


class SpatialGatingUnit(hk.Module):
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


class MLPMixerXPatchFeedForward(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        dim_hidden: Optional[int] = None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.dim_hidden = num_patches * 4 if dim_hidden is None else dim_hidden
        self.activation = activation

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = hk.Conv1D(self.dim_hidden, 1, data_format="NCW", name="conv_1")(inputs)
        x = self.activation(x)
        return hk.Conv1D(self.num_patches, 1, data_format="NCW", name="conv_2")(x)


class ResMLPXPatchFeedForward(hk.Module):
    def __init__(self, num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return hk.Conv1D(self.num_patches, 1, data_format="NCW", name="conv")(inputs)


class gMLPFeedForward(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        dim_hidden: Optional[int] = None,
        sgu: Optional[XModuleFactory] = None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        sgu_kwargs, kwargs = group_by_prefix_and_trim("sgu_", kwargs)

        if sgu is None:
            sgu = sgu_factory

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.dim_hidden = dim * 4 if dim_hidden is None else dim_hidden
        self.sgu = sgu
        self.activation = activation
        self.sgu_kwargs = sgu_kwargs

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.dim_hidden, name="proj_hidden")(inputs)
        x = self.activation(x)
        x = self.sgu(self.num_patches, self.dim, self.depth, **self.sgu_kwargs, name="sgu")(x)
        x = hk.Linear(self.dim, name="proj_dim")(x)
        return x


class XChannelFeedForward(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        dim_hidden: Optional[int] = None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        shift: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.dim_hidden = dim * 4 if dim_hidden is None else dim_hidden
        self.activation = activation
        self.shift = shift

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.dim_hidden, name="linear_1")(inputs)
        x = self.activation(x)
        if self.shift is not None:
            x = self.shift(x)
        x = hk.Linear(self.dim, name="linear_2")(x)
        return x


class XSublayer(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        ff: XModuleFactory,
        prenorm: Optional[XModuleFactory] = None,
        postnorm: Optional[XModuleFactory] = None,
        residual: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        ff_kwargs, kwargs = group_by_prefix_and_trim("ff_", kwargs)
        prenorm_kwargs, kwargs = group_by_prefix_and_trim("prenorm_", kwargs)
        postnorm_kwargs, kwargs = group_by_prefix_and_trim("postnorm_", kwargs)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.ff = ff
        self.prenorm = prenorm
        self.postnorm = postnorm
        self.residual = residual
        self.ff_kwargs = ff_kwargs
        self.prenorm_kwargs = prenorm_kwargs
        self.postnorm_kwargs = postnorm_kwargs

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        if self.prenorm is not None:
            x = self.prenorm(self.num_patches, self.dim, self.depth, **self.prenorm_kwargs)(x)
        x = self.ff(self.num_patches, self.dim, self.depth, **self.ff_kwargs)(x)
        if self.postnorm is not None:
            x = self.postnorm(self.num_patches, self.dim, self.depth, **self.postnorm_kwargs)(x)
        if self.residual:
            x += inputs

        return x


class XBlock(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        sublayers: Sequence[XModuleFactory],
        residual: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        sublayer_common_kwargs, kwargs = group_by_prefix_and_trim("sublayers_", kwargs)
        sublayers_kwargs = []
        for i in range(len(sublayers)):
            sublayer_kwargs, kwargs = group_by_prefix_and_trim(f"sublayer{i + 1}_", kwargs)
            sublayers_kwargs.append(sublayer_kwargs)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.sublayers = tuple(sublayers)
        self.residual = residual
        self.sublayer_common_kwargs = sublayer_common_kwargs
        self.sublayers_kwargs = sublayers_kwargs

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for i, sublayer in enumerate(self.sublayers):
            sublayer_kwargs = self.sublayer_common_kwargs.copy()
            sublayer_kwargs.update(self.sublayers_kwargs[i])
            x = sublayer(self.num_patches, self.dim, self.depth, **sublayer_kwargs)(x)
        if self.residual:
            x += inputs
        return x


class XMLP(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        block: XModuleFactory,
        normalization: Optional[XModuleFactory] = None,
        num_classes: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        block_kwargs, kwargs = group_by_prefix_and_trim("block_", kwargs)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.block = block
        self.normalization = normalization
        self.num_classes = num_classes
        self.block_kwargs = block_kwargs

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


def layernorm_factory(num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
    axis, create_scale, create_offset = pick_and_pop(["axis", "create_scale", "create_offset"], kwargs)
    if axis is None:
        axis = -1
    if create_scale is None:
        create_scale = True
    if create_offset is None:
        create_offset = True

    return hk.LayerNorm(axis, create_scale, create_offset, **kwargs, name=name)


def sgu_factory(num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
    return SpatialGatingUnit(num_patches, dim, depth, **kwargs, name=name)


def gmlp_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    return gMLPFeedForward(num_patches, dim, depth, **kwargs, name=name)


def mlpmixer_xpatch_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    return MLPMixerXPatchFeedForward(num_patches, dim, depth, **kwargs, name=name)


def resmlp_xpatch_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    return ResMLPXPatchFeedForward(num_patches, dim, depth, **kwargs, name=name)


def xchannel_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    return XChannelFeedForward(num_patches, dim, depth, **kwargs, name=name)


def gmlp_block_factory(num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
    return XBlock(
        num_patches,
        dim,
        depth,
        [
            lambda num_patches, dim, depth, **kwargs: XSublayer(
                num_patches, dim, depth, ff=gmlp_feedforward_factory, prenorm=layernorm_factory, **kwargs
            )
        ],
        name=name,
        **kwargs,
    )


def mlpmixer_block_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    return XBlock(
        num_patches,
        dim,
        depth,
        [
            # Cross patch sublayer
            lambda num_patches, dim, depth, **kwargs: XSublayer(
                num_patches, dim, depth, ff=mlpmixer_xpatch_feedforward_factory, prenorm=layernorm_factory, **kwargs
            ),
            # Cross channel sublayer
            lambda num_patches, dim, depth, **kwargs: XSublayer(
                num_patches, dim, depth, ff=xchannel_feedforward_factory, prenorm=layernorm_factory, **kwargs
            ),
        ],
        name=name,
        **kwargs,
    )


def resmlp_block_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    return XBlock(
        num_patches,
        dim,
        depth,
        [
            lambda num_patches, dim, depth, **kwargs: XSublayer(
                num_patches,
                dim,
                depth,
                ff=resmlp_xpatch_feedforward_factory,
                prenorm=lambda num_patches, dim, depth, **kwargs: Affine(dim, **kwargs),
                postnorm=lambda num_patches, dim, depth, **kwargs: LayerScale(dim, depth, **kwargs),
                **kwargs,
            ),
            lambda num_patches, dim, depth, **kwargs: XSublayer(
                num_patches,
                dim,
                depth,
                ff=xchannel_feedforward_factory,
                prenorm=lambda num_patches, dim, depth, **kwargs: Affine(dim, **kwargs),
                postnorm=lambda num_patches, dim, depth, **kwargs: LayerScale(dim, depth, **kwargs),
                **kwargs,
            ),
        ],
        name=name,
        **kwargs,
    )


def s2mlp_block_factory(num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
    if "sublayer1_ff_shift" not in kwargs:
        raise ValueError("s2mlp_block_factory requires sublayer1_ff_shift to be specified")
    if "sublayer1_ff_dim_hidden" not in kwargs and "sublayers_ff_dim_hidden" not in kwargs:
        kwargs["sublayer1_ff_dim_hidden"] = dim

    return XBlock(
        num_patches,
        dim,
        depth,
        [
            # Cross patch sublayer
            lambda num_patches, dim, depth, **kwargs: XSublayer(
                num_patches, dim, depth, ff=xchannel_feedforward_factory, prenorm=layernorm_factory, **kwargs
            ),
            # Cross channel sublayer
            lambda num_patches, dim, depth, **kwargs: XSublayer(
                num_patches, dim, depth, ff=xchannel_feedforward_factory, prenorm=layernorm_factory, **kwargs
            ),
        ],
        name=name,
        **kwargs,
    )


__all__ = [
    "Affine",
    "LayerScale",
    "MLPMixerXPatchFeedForward",
    "ResMLPXPatchFeedForward",
    "SpatialGatingUnit",
    "XBlock",
    "XChannelFeedForward",
    "XMLP",
    "XSublayer",
    "create_shift2d_op",
    "gMLPFeedForward",
    "gmlp_block_factory",
    "gmlp_feedforward_factory",
    "layernorm_factory",
    "mlpmixer_block_factory",
    "mlpmixer_xpatch_feedforward_factory",
    "resmlp_block_factory",
    "resmlp_xpatch_feedforward_factory",
    "s2mlp_block_factory",
    "sgu_factory",
    "xchannel_feedforward_factory",
]
