from typing import Any, Callable, Optional, Protocol, Sequence, Union

import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange, reduce

from ._utils import group_by_prefix_and_trim, pick_and_pop


class XModuleFactory(Protocol):
    """Defines a common factory function interface for all X-MLP modules."""

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


__all__ = [
    "Affine",
    "LayerScale",
    "MLPMixerXPatchFeedForward",
    "ResMLPXPatchFeedForward",
    "SampleDropout",
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
