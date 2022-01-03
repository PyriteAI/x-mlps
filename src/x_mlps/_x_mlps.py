from typing import Any, Optional, Sequence, Union

import haiku as hk
import jax.numpy as jnp
from einops import reduce

from ._ff import (
    fft_xpatch_feedforward_factory,
    fftlinear_xpatch_feedforward_factory,
    gmlp_feedforward_factory,
    mlpmixer_xpatch_feedforward_factory,
    resmlp_xpatch_feedforward_factory,
    xchannel_feedforward_factory,
)
from ._nn import Affine, LayerScale, SampleDropout, layernorm_factory
from ._types import XModuleFactory
from ._utils import group_by_prefix_and_trim


class XSublayer(hk.Module):
    """Flexible sublayer wrapper module providing skip connections and pre/post-normalization to arbitrary layers.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        ff (XModuleFactory): Feedforward layer factory function.
        prenorm (XModuleFactory, optional): Pre-normalization layer factory function. Defaults to `None`.
        postnorm (XModuleFactory, optional): Post-normalization layer factory function. Defaults to `None`.
        residual (bool): Whether to add a residual/skip connection. Defaults to `True`.
        drop_path_survival_rate (float): Probability of the core computation being active (not dropped). Only applicable
            if `residual` is `True`. Defaults to 1.0.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: All arguments starting with "ff_" are passed to the feedforward layer factory function.
            All arguments starting with "prenorm_" are passed to the pre-normalization layer factory function.
            All arguments starting with "postnorm_" are passed to the post-normalization layer factory function.
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        ff: XModuleFactory,
        prenorm: Optional[XModuleFactory] = None,
        postnorm: Optional[XModuleFactory] = None,
        residual: bool = True,
        drop_path_survival_rate: float = 1.0,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        ff_kwargs, kwargs = group_by_prefix_and_trim("ff_", kwargs)
        prenorm_kwargs, kwargs = group_by_prefix_and_trim("prenorm_", kwargs)
        postnorm_kwargs, kwargs = group_by_prefix_and_trim("postnorm_", kwargs)
        if kwargs:
            raise KeyError(f"unknown keyword arguments: {list(kwargs.keys())}")

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.ff = ff
        self.prenorm = prenorm
        self.postnorm = postnorm
        self.residual = residual
        self.drop_path_survival_rate = drop_path_survival_rate
        self.ff_kwargs = ff_kwargs
        self.prenorm_kwargs = prenorm_kwargs
        self.postnorm_kwargs = postnorm_kwargs

    def __call__(self, inputs: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        """Propogate inputs through the sublayer.

        Args:
            inputs (jnp.ndarray): Inputs to the sublayer.
            is_training (bool): If `True`, enable training specific features (e.g., dropout). Keyword argument.

        Returns:
            jnp.ndarray: Outputs of the sublayer.
        """
        x = inputs
        if self.prenorm is not None:
            x = self.prenorm(self.num_patches, self.dim, self.depth, **self.prenorm_kwargs)(x)
        x = self.ff(self.num_patches, self.dim, self.depth, **self.ff_kwargs)(x)
        if self.postnorm is not None:
            x = self.postnorm(self.num_patches, self.dim, self.depth, **self.postnorm_kwargs)(x)
        if self.residual:
            x = SampleDropout(1 - self.drop_path_survival_rate)(x, is_training=is_training) + inputs

        return x


class XBlock(hk.Module):
    """Generic MLP block.

    One or more `XSublayer` modules are stacked together to form a block. Optionally, a skip connection can be added.
    Arbitrary arguments can be passed to `XSublayer` modules two different ways:

    1. As keyword arguments prefixed with "sublayers_". These arguments are passed to all the sublayers.
    2. As keyword arguments prefixed with "sublayer{i}_" where 1 <= i <= len(sublayers). These arguments are passed to
        to the i-th sublayer.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of this block in the network. Note that depth starts from 1.
        sublayers (Sequence[XSublayerFactory]): Sublayer factory functions. Created sublayers will be stacked in the
            order of their respective factory function in the sequence.
        residual (bool): Whether to add a residual/skip connection. Defaults to `False`.
        drop_path_survival_rate (float): Probability of the core computation being active (not dropped). Passed directly
            to sublayers. This will also be applied at the block level if residual is `True`. Defaults to 1.0.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: All arguments starting with "sublayers_" are passed to all sublayers. All arguments starting with
            "sublayer{i}_" are passed to the i-th sublayer.
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        sublayers: Sequence[XModuleFactory],
        residual: bool = False,
        drop_path_survival_rate: float = 1.0,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        sublayer_common_kwargs, kwargs = group_by_prefix_and_trim("sublayers_", kwargs)
        sublayers_kwargs = []
        for i in range(len(sublayers)):
            sublayer_kwargs, kwargs = group_by_prefix_and_trim(f"sublayer{i + 1}_", kwargs)
            sublayers_kwargs.append(sublayer_kwargs)
        if kwargs:
            raise KeyError(f"unknown keyword arguments: {list(kwargs.keys())}")

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.sublayers = tuple(sublayers)
        self.residual = residual
        self.drop_path_survival_rate = drop_path_survival_rate
        self.sublayer_common_kwargs = sublayer_common_kwargs
        self.sublayers_kwargs = sublayers_kwargs

    def __call__(self, inputs: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        """Propogate inputs through the block.

        Args:
            inputs (jnp.ndarray): Inputs to the block.
            is_training (bool): If `True`, enable training specific features (e.g., dropout). Keyword argument.

        Returns:
            jnp.ndarray: Outputs of the block.
        """
        x = inputs
        for i, sublayer in enumerate(self.sublayers):
            sublayer_kwargs = self.sublayer_common_kwargs.copy()
            sublayer_kwargs.update(self.sublayers_kwargs[i])
            x = sublayer(
                self.num_patches,
                self.dim,
                self.depth,
                drop_path_survival_rate=self.drop_path_survival_rate,
                **sublayer_kwargs,
            )(x, is_training=is_training)
        if self.residual:
            x = SampleDropout(1 - self.drop_path_survival_rate)(x, is_training=is_training) + inputs
        return x


class XMLP(hk.Module):
    """Generic MLP network.

    N `XBlock` modules are stacked together to form a network (where N is set to `depth`). Importantly, this network
    assumes the input has been formatted appropriately (e.g., a sequence of patches). Before data is processed by the
    stack of `XBlock` modules, it is first projected to the specified dimension `dim` via a linear layer.

    This network can optionally be configured with Stochastic Depth¹, a form of regularization. If enabled, the depth
    of the network will be dynamically adjusted during training, with sections of the network being randomly dropped.
    The likelihood of dropping a layer can either fixed, or dependent on the depth of the network.

    Optionally, the network can be configured to have a classification layer at the end by setting `num_classes` to a
    non-zero value. In this case, the resulting sequence from stack of `XBlock` modules will be averaged over the
    sequence dimension before being fed to the classification layer.

    Arbitrary arguments can be passed to `XBlock` modules by prepending the argument name with "block_". Further, to
    ensure arguments are passed to child modules of each `XBlock` module, each argument name should additionally be
    prefixed with that child module's identifier, starting with "block" and working down the hierarchy. For example,
    to pass an argument to the feedforward module of the first sublayer of each block, the argument name should be
    "block_sublayer1_ff_<argument name>".

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension. Inputs fed to this network are projected to this dimension.
        depth (int): The number of blocks in the network.
        block (XBlockFactory): Block factory function.
        normalization (XModuleFactory, optional): Normalization module factory function. Occurs after the stack of
            `XBlock` modules. Useful for pre-normalization architectures. Defaults to None.
        stochastic_depth (Union[bool, float], optional): Whether to use stochastic depth. If `True`, the surivival rate
            of each block follows the linear decay function 1 - 0.5 * (i / depth) for 1 <= i <= depth. If `False`, the
            survival rate is 1.0. If a float, the survival rate is set to this value. Defaults to False.
        num_classes (int, optional): Number of classes in the classification layer. Defaults to None.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: All arguments starting with "block_" are passed to all blocks.

    References:
        1. Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382).
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        depth: int,
        block: XModuleFactory,
        normalization: Optional[XModuleFactory] = None,
        stochastic_depth: Union[float, bool] = False,
        num_classes: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)

        block_kwargs, kwargs = group_by_prefix_and_trim("block_", kwargs)
        if kwargs:
            raise KeyError(f"unknown keyword arguments: {list(kwargs.keys())}")

        if isinstance(stochastic_depth, bool) and stochastic_depth:
            # This ensures that the first block can be dropped as well.
            drop_path_survival_rates = jnp.linspace(1.0, 0.5, num=depth + 1)[1:]
        elif isinstance(stochastic_depth, float):
            drop_path_survival_rates = jnp.full(depth, stochastic_depth)
        else:
            drop_path_survival_rates = jnp.ones(depth)

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.block = block
        self.normalization = normalization
        self.drop_path_survival_rates = drop_path_survival_rates
        self.num_classes = num_classes
        self.block_kwargs = block_kwargs

    def __call__(self, inputs: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        """Propogate inputs through the network.

        Args:
            inputs (jnp.ndarray): Inputs to the network.
            is_training (bool): If `True`, enable training specific features (e.g., dropout). Keyword argument.

        Returns:
            jnp.ndarray: Outputs of the network.
        """
        x = hk.Linear(self.dim, name="proj_in")(inputs)
        for i in range(self.depth):
            x = self.block(
                self.num_patches,
                self.dim,
                i + 1,
                drop_path_survival_rate=self.drop_path_survival_rates[i],
                **self.block_kwargs,
            )(x, is_training=is_training)
        if self.normalization is not None:
            x = self.normalization(self.num_patches, self.dim, self.depth + 1)(x)
        if self.num_classes is not None:
            x = reduce(x, "... n c -> ... c", "mean")
            x = hk.Linear(self.num_classes, name="proj_out")(x)
        return x


def gmlp_block_factory(num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
    """gMLP block module factory function.

    Builds a `XBlock` module with the gMLP block structure as defined in *Pay Attention to MLPs*¹. Specifically, this
    consists of a single `XSublayer` with a `gMLPFeedForward` module and layer normalization (pre-normalization).

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional block and child module arguments.

    Returns:
        hk.Module: `XBlock` module.

    References:
        1. Pay Attention to MLPs (https://arxiv.org/abs/2105.08050).
    """
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
    """MLP-Mixer block module factory function.

    Builds a `XBlock` module with the MLP-Mixer block structure as defined in *MLP-Mixer: An all-MLP Architecture for
    Vision*¹. Specifically, this consists of two `XSublayer`s: 1) a `MLPMixerXPatchFeedForward` module and 2) a
    `XChannelFeedForward` module. Both make use of layer normalization (pre-normalization).

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional block and child module arguments.

    Returns:
        hk.Module: `XBlock` module.

    References:
        1. MLP-Mixer: An all-MLP Architecture for Vision (https://arxiv.org/abs/2105.01601).
    """
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
    """ResMLP block module factory function.

    Builds a `XBlock` module with the ResMLP block structure as defined in *ResMLP: Feedforward networks for image
    classification with data-efficient training*¹. Specifically, this consists of two `XSublayer`s: 1) a
    `ResMLPXPatchFeedForward` module and 2) a `XChannelFeedForward` module. Both make use of `Affine` pre-normalization
    and `LayerScale` post-normalization.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional block and child module arguments.

    Returns:
        hk.Module: `XBlock` module.

    References:
        1. ResMLP: Feedforward networks for image classification with data-efficient training
            (https://arxiv.org/abs/2105.03404).
    """
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
    """S^2-MLP block module factory function.

    Builds a `XBlock` module with the S^2-MLP block structure as defined in *S^2-MLP: Spatial-Shift MLP Architecture
    for Vision*¹. Specifically, this consists of two `XSublayer`s: 1) a `XChannelFeedForward` module with a shift
    function and 2) a second `XChannelFeedForward` module. Both make use of layer normalization (pre-normalization).

    Note: currently, the spatial shift function must be passed as the keyword argument "sublayer1_ff_shift". This is
    due to the fact that `create_shift2d_op` requires post-patching height and width dimensions, which the `XMLP`
    module has no knowledge of.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional block and child module arguments.

    Returns:
        hk.Module: `XBlock` module.

    References:
        1. S2-MLP: Spatial-Shift MLP Architecture for Vision (https://arxiv.org/abs/2106.07477).
    """

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


def fft_block_factory(num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
    """FFT block module factory function.

    Builds a `XBlock` module with an FNet-like block structure as defined in *FNet: Mixing Tokens with Fourier Transforms*¹.
    Specifically, this consists of two `XSublayer`s: 1) an `FFTFeedForward` module and 2) a `XChannelFeedForward` module.
    Both make use of layer normalization (pre-normalization).

    In contrast to the FNet paper, the FFT feedforward layer can be configured to an arbitrary N-D FFT. By default it is
    configured to be a 1D FFT across the patch (token) dimension.

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional block and child module arguments.

    Returns:
        hk.Module: `XBlock` module.

    References:
        1. FNet: Mixing Tokens with Fourier Transforms (https://arxiv.org/abs/2105.03824).
    """
    return XBlock(
        num_patches,
        dim,
        depth,
        [
            # Cross patch sublayer
            lambda num_patches, dim, depth, name=None, **kwargs: XSublayer(
                num_patches,
                dim,
                depth,
                ff=fft_xpatch_feedforward_factory,
                prenorm=layernorm_factory,
                name=name,
                **kwargs,
            ),
            # Cross channel sublayer
            lambda num_patches, dim, depth, name=None, **kwargs: XSublayer(
                num_patches, dim, depth, ff=xchannel_feedforward_factory, prenorm=layernorm_factory, name=name, **kwargs
            ),
        ],
        name=name,
        **kwargs,
    )


def fftlinear_block_factory(
    num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any
) -> hk.Module:
    """FFT-Linear block module factory function.

    Builds a `XBlock` module with an FNet-like block structure as defined in *FNet: Mixing Tokens with Fourier Transforms*¹.
    Specifically, this consists of two `XSublayer`s: 1) an `FFTLinearFeedForward` module and 2) a `Linear` module. Both
    make use of layer normalization (pre-normalization).

    In contrast to the FNet paper, the feedforward layer consists of a 1-D FFT across the patch (token) dimension and a
    linear layer (which consumes the FFT'ed data).

    Satifies the `XModuleFactory` interface.

    Args:
        num_patches (int): Number of patches in the input.
        dim (int): Size of the channel dimension.
        depth (int): The depth of the block which contains this layer in the network. Note that depth starts from 1.
        name (str, optional): The name of the module. Defaults to None.
        **kwargs: Additional block and child module arguments.

    Returns:
        hk.Module: `XBlock` module.

    References:
        1. FNet: Mixing Tokens with Fourier Transforms (https://arxiv.org/abs/2105.03824).
    """
    return XBlock(
        num_patches,
        dim,
        depth,
        [
            # Cross patch sublayer
            lambda num_patches, dim, depth, name=None, **kwargs: XSublayer(
                num_patches,
                dim,
                depth,
                ff=fftlinear_xpatch_feedforward_factory,
                prenorm=layernorm_factory,
                name=name,
                **kwargs,
            ),
            # Linear sublayer
            lambda num_patches, dim, depth, name=None, **kwargs: XSublayer(
                num_patches, dim, depth, ff=xchannel_feedforward_factory, prenorm=layernorm_factory, name=name, **kwargs
            ),
        ],
        name=name,
        **kwargs,
    )


__all__ = [
    "XBlock",
    "XMLP",
    "XSublayer",
    "fft_block_factory",
    "fftlinear_block_factory",
    "gmlp_block_factory",
    "mlpmixer_block_factory",
    "resmlp_block_factory",
    "s2mlp_block_factory",
]
