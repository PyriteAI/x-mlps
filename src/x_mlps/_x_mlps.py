from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from einops import reduce


class PreAffine(hk.Module):
    def __init__(self, dim: int, name: Optional[str] = None):
        super().__init__(name=name)

        self.dim = dim

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = (1,) * (inputs.ndim - 1) + (self.dim,)
        w = hk.get_parameter("w", shape=shape, init=jnp.ones)
        b = hk.get_parameter("b", shape=shape, init=jnp.zeros)

        return w * inputs + b


class PostAffine(hk.Module):
    def __init__(self, dim: int, depth: int, name: Optional[str] = None):
        super().__init__(name=name)

        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.dim = dim
        self.init_eps = init_eps

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = (1,) * (inputs.ndim - 1) + (self.dim,)
        w = hk.get_parameter("w", shape=shape, init=hk.initializers.Constant(self.init_eps))
        b = hk.get_parameter("b", shape=shape, init=jnp.zeros)

        return w * inputs + b


class ResMLPXPatchSublayer(hk.Module):
    def __init__(self, num_patches: int, dim: int, depth: int, name: Optional[str] = None):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = PreAffine(self.dim, name="affine")(inputs)
        x = hk.Conv1D(self.num_patches, 1, data_format="NCW", name="conv")(x)
        x = PostAffine(self.dim, self.depth, name="scale")(x)

        return x + inputs


class ResMLPXChannelSublayer(hk.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        dim_hidden: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.dim = dim
        self.depth = depth
        self.activation = activation
        self.dim_hidden = dim * 4 if dim_hidden is None else dim_hidden

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = PreAffine(self.dim, name="affine")(inputs)
        x = hk.Linear(self.dim_hidden, name="linear_1")(x)
        x = self.activation(x)
        x = hk.Linear(self.dim, name="linear_2")(x)
        x = PostAffine(self.dim, self.depth, name="scale")(x)

        return x + inputs


class XMLP(hk.Module):
    def __init__(
        self, num_patches: int, dim: int, depth: int, num_classes: Optional[int] = None, name: Optional[str] = None
    ):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.dim, name="proj_in")(inputs)
        for i in range(self.depth):
            x = ResMLPXPatchSublayer(self.num_patches, self.dim, depth=i + 1, name=f"x_patch_sublayer_{i}")(x)
            x = ResMLPXChannelSublayer(self.dim, depth=i + 1, name=f"x_channel_sublayer_{i}")(x)
        x = PreAffine(self.dim, name="affine")(x)
        if self.num_classes is not None:
            x = reduce(x, "... n c -> ... c", "mean")
            x = hk.Linear(self.num_classes, name="proj_out")(x)
        return x


__all__ = ["PostAffine", "PreAffine", "XMLP"]
