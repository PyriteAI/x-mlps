from typing import Any, Callable, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

from ._nn import sgu_factory
from ._types import XModuleFactory
from ._utils import group_by_prefix_and_trim


class MLPMixerXPatchFeedForward(hk.Module):
    """Patch (token) mixing feedforward layer as described in *MLP-Mixer: An all-MLP Architecture for Vision*¹.

    Note that this module does not implement normalization nor a skip connection. This module can be combined with the
    `XSublayer` module to add these functionalities.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension. Not used directly, rather it's included as an argument to establish
            a consistent interface with other modules.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        dim_hidden (int, optional): Hidden dimension size. Defaults to 4 x num_patches.
        activation (Callable[[jnp.ndarray], jnp.ndarray], optional): Activation function. Defaults to the GELU function.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Currently unused.

    References:
        1. MLP-Mixer: An all-MLP Architecture for Vision (https://arxiv.org/abs/2105.01601).
    """

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

        if kwargs:
            raise KeyError(f"unknown keyword arguments: {list(kwargs.keys())}")

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.dim_hidden = num_patches * 4 if dim_hidden is None else dim_hidden
        self.activation = activation

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # NOTE: Equivalent to a transpose and pair of linear layers.
        x = hk.Conv1D(self.dim_hidden, 1, data_format="NCW", name="conv_1")(inputs)
        x = self.activation(x)
        return hk.Conv1D(self.num_patches, 1, data_format="NCW", name="conv_2")(x)


class ResMLPXPatchFeedForward(hk.Module):
    """Patch (token) mixing feedforward layer as described in ResMLP¹.

    Note that this module does not implement normalization nor a skip connection. This module can be combined with the
    `XSublayer` module to add these functionalities.

    Args
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension. Not used directly, rather it's included as an argument to establish
            a consistent interface with other modules.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        **kwargs: Currently unused.

    References:
        1. ResMLP: Feedforward networks for image classification with data-efficient training
            (https://arxiv.org/abs/2105.03404).
    """

    def __init__(self, num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any):
        super().__init__(name=name)

        if kwargs:
            raise KeyError(f"unknown keyword arguments: {list(kwargs.keys())}")

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # NOTE: Equivalent to a transpose and linear layer.
        return hk.Conv1D(self.num_patches, 1, data_format="NCW", name="conv")(inputs)


class gMLPFeedForward(hk.Module):
    """gMLP feedforward layer as described in *Pay Attention to MLPs*¹.

    Note that this module does not implement normalization nor a skip connection. This module can be combined with the
    `XSublayer` module to add these functionalities.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        dim_hidden (int, optional): Hidden dimension size of the first projection. Defaults to 4 x dim.
        sgu (XModuleFactory, optional): Spatial gating unit factory function. Defaults to the standard SpatialGatingUnit
            module via `sgu_factory`.
        activation (Callable[[jnp.ndarray], jnp.ndarray], optional): Activation function. Defaults to the GELU function.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: All arguments starting with "sgu_" are passed to the spatial gating unit factory function.

    References:
        1. Pay Attention to MLPs (https://arxiv.org/abs/2105.08050).
    """

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
        if kwargs:
            raise KeyError(f"unknown keyword arguments: {list(kwargs.keys())}")

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


class FFTFeedForward(hk.Module):
    """N-D FFT mixing feedforward layer.

    Mixes patches (tokens) using an N-D FFT. By default this is done in the patch/token/sequence dimension using a 1D FFT.
    Returns only the real component of the FFT.

    Note that this module does not implement normalization nor a skip connection. This module can be combined with the
    `XSublayer` module to add these functionalities.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        axes (Sequence[int]): The axes to perform the FFT on. Defaults to (-2,).
        norm (str, optional): The normalization to apply to the FFT. Defaults to None.
        name (str, optional): The name of the module. Defaults to None.
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        axes: Sequence[int] = (-2,),
        norm: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.axes = axes
        self.norm = norm

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.fftn(inputs, axes=self.axes, norm=self.norm).real


class FFTLinearFeedForward(hk.Module):
    """1-D FFT mixing combined with a linear layer.

    Mixes patches (tokens) using an 1-D FFT across the patch/token/sequence dimension and then feeds the results to a
    linear layer. Operates only on the real component of the FFT.

    Note that this module does not implement normalization nor a skip connection. This module can be combined with the
    `XSublayer` module to add these functionalities.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        axis (int): The axes to perform the FFT on. Defaults to (-2,).
        norm (str, optional): The normalization to apply to the FFT. Defaults to None.
        name (str, optional): The name of the module. Defaults to None.
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        axis: int = -2,
        norm: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.axis = axis
        self.norm = norm

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = jnp.fft.fft(inputs, axis=self.axis, norm=self.norm).real
        return hk.Linear(self.dim, name="linear")(x)


class XChannelFeedForward(hk.Module):
    """Common channel mixing feedforward layer.

    Note that this module does not implement normalization nor a skip connection. This module can be combined with the
    `XSublayer` module to add these functionalities.

    Args:
        num_patches (int): Number of patches in the input. Not used directly, rather it's included as an argument to
            establish a consistent interface with other modules.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        dim_hidden (int, optional): Hidden dimension size. Defaults to 4 x dim.
        activation (Callable[[jnp.ndarray], jnp.ndarray], optional): Activation function. Defaults to the GELU function.
        shift (Callable[[jnp.ndarray], jnp.ndarray], optional): Token shifting function (e.g., spatial shift in S^2-MLP).
            Defaults to `None`.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Currently unused.
    """

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

        if kwargs:
            raise KeyError(f"unknown keyword arguments: {list(kwargs.keys())}")

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


def gmlp_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    """gMLP feedforward module factory function.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional `gMLPFeedForward` arguments.

    Returns:
        hk.Module: `gMLPFeedForward` module.
    """

    return gMLPFeedForward(num_patches, dim, depth, **kwargs, name=name)


def mlpmixer_xpatch_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    """MLP-Mixer cross-patch feedforward module factory function.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional `MLPMixerXPatchFeedForward` arguments.

    Returns:
        hk.Module: `MLPMixerXPatchFeedForward` module.
    """

    return MLPMixerXPatchFeedForward(num_patches, dim, depth, **kwargs, name=name)


def resmlp_xpatch_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    """ResMLP cross-patch feedforward module factory function.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional `ResMLPXPatchFeedForward` arguments.

    Returns:
        hk.Module: `ResMLPXPatchFeedForward` module.
    """
    return ResMLPXPatchFeedForward(num_patches, dim, depth, **kwargs, name=name)


def fft_xpatch_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    """FFT cross-patch feedforward module factory function.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional `FFTFeedForward` arguments.

    Returns:
        hk.Module: `FFTFeedForward` module.
    """
    return FFTFeedForward(num_patches, dim, depth, **kwargs, name=name)


def fftlinear_xpatch_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    """FFT-Linear cross-patch feedforward module factory function.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional `FFTLinearFeedForward` arguments.

    Returns:
        hk.Module: `FFTLinearFeedForward` module.
    """
    return FFTLinearFeedForward(num_patches, dim, depth, **kwargs, name=name)


def xchannel_feedforward_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    """Standard cross-channel feedforward module factory function.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional `XChannelFeedForward` arguments.

    Returns:
        hk.Module: `XChannelFeedForward` module.
    """
    return XChannelFeedForward(num_patches, dim, depth, **kwargs, name=name)


__all__ = [
    "MLPMixerXPatchFeedForward",
    "FFTFeedForward",
    "FFTLinearFeedForward",
    "ResMLPXPatchFeedForward",
    "XChannelFeedForward",
    "fft_xpatch_feedforward_factory",
    "fftlinear_xpatch_feedforward_factory",
    "gMLPFeedForward",
    "gmlp_feedforward_factory",
    "mlpmixer_xpatch_feedforward_factory",
    "resmlp_xpatch_feedforward_factory",
    "xchannel_feedforward_factory",
]
