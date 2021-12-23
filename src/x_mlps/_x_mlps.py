from typing import Callable, Literal, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from einops import reduce

from ._utils import group_by_prefix_and_trim

FFStrategy = Literal["mlpmixer", "resmlp"]


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


class MLPMixerFeedForward(hk.Module):
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


class ResMLPFeedForward(hk.Module):
    def __init__(self, dim: int, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)

        self.dim = dim

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return hk.Conv1D(self.dim, 1, data_format="NCW", name="conv")(inputs)


class XPatchSublayer(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        normalization: Optional[Callable[[int, str], hk.Module]] = None,
        feedforward: FFStrategy = "mlpmixer",
        layer_affine: bool = False,
        layer_scale: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)

        if feedforward not in ["mlpmixer", "resmlp"]:
            raise ValueError(f"unknown feedforward type: '{feedforward}'")
        if normalization is None:
            normalization = lambda dim, name: hk.LayerNorm(  # noqa: E731
                -1, create_scale=True, create_offset=True, name=name
            )
        ff_kwargs, kwargs = group_by_prefix_and_trim("ff_", kwargs)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.prenorm = normalization
        if feedforward == "mlpmixer":
            self.ff = MLPMixerFeedForward(self.num_patches, **ff_kwargs, name="ff")
        elif feedforward == "resmlp":
            self.ff = ResMLPFeedForward(self.num_patches, **ff_kwargs, name="ff")
        if layer_affine:
            self.postnorm = Affine(self.dim, init_scale=_calc_layer_scale_eps(depth), name="affine")
        elif layer_scale:
            self.postnorm = LayerScale(self.dim, self.depth, name="scale")
        else:
            self.postnorm = None

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = self.prenorm(self.dim, name="norm")(inputs)
        x = self.ff(x)
        if self.postnorm is not None:
            x = self.postnorm(x)

        return x + inputs


class XChannelSublayer(hk.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_hidden: Optional[int] = None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        normalization: Optional[Callable[[int, str], hk.Module]] = None,
        layer_affine: bool = False,
        layer_scale: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if layer_affine and layer_scale:
            raise ValueError("cannot have both layer_affine and layer_scale set to True")

        if normalization is None:
            normalization = lambda dim, name: hk.LayerNorm(  # noqa: E731
                -1, create_scale=True, create_offset=True, name=name
            )

        self.dim = dim
        self.depth = depth
        self.dim_hidden = dim * 4 if dim_hidden is None else dim_hidden
        self.activation = activation
        self.prenorm = normalization
        if layer_affine:
            self.postnorm = Affine(self.dim, init_scale=_calc_layer_scale_eps(depth), name="affine")
        elif layer_scale:
            self.postnorm = LayerScale(self.dim, self.depth, name="scale")
        else:
            self.postnorm = None

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = self.prenorm(self.dim, name="norm")(inputs)
        x = hk.Linear(self.dim_hidden, name="linear_1")(x)
        x = self.activation(x)
        x = hk.Linear(self.dim, name="linear_2")(x)
        if self.postnorm is not None:
            x = self.postnorm(x)

        return x + inputs


class XMLP(hk.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        num_classes: Optional[int] = None,
        normalization: Optional[Callable[[int, str], hk.Module]] = None,
        patch_feedforward: FFStrategy = "mlpmixer",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)

        if normalization is None:
            normalization = lambda dim, name: hk.LayerNorm(  # noqa: E731
                -1, create_scale=True, create_offset=True, name=name
            )

        patch_kwargs, kwargs = group_by_prefix_and_trim("patch_", kwargs)
        channel_kwargs, kwargs = group_by_prefix_and_trim("channel_", kwargs)
        patch_kwargs["feedforward"] = patch_feedforward
        patch_kwargs.pop("name", None)
        channel_kwargs.pop("name", None)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes
        self.normalization = normalization
        self.patch_kwargs = patch_kwargs
        self.channel_kwargs = channel_kwargs

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.dim, name="proj_in")(inputs)
        for i in range(self.depth):
            x = XPatchSublayer(
                self.num_patches,
                self.dim,
                depth=i + 1,
                **self.patch_kwargs,
                name=f"x_patch_sublayer_{i + 1}",
            )(x)
            x = XChannelSublayer(self.dim, depth=i + 1, **self.channel_kwargs, name=f"x_channel_sublayer_{i + 1}")(x)
        x = self.normalization(self.dim, name="norm")(x)
        if self.num_classes is not None:
            x = reduce(x, "... n c -> ... c", "mean")
            x = hk.Linear(self.num_classes, name="proj_out")(x)
        return x


__all__ = [
    "Affine",
    "LayerScale",
    "MLPMixerFeedForward",
    "ResMLPFeedForward",
    "XChannelSublayer",
    "XMLP",
    "XPatchSublayer",
]
