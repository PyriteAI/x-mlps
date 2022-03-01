# X-MLPs

An MLP model that provides a flexible foundation to implement, mix-and-match, and test various state-of-the-art MLP building blocks and architectures.
Built on JAX and Haiku.

## Installation

```sh
pip install x-mlps
```

**Note**: X-MLPs will not install JAX for you (see [here](https://github.com/google/jax#installation) for install instructions).

## Getting Started

The `XMLP` class provides the foundation from which all MLP architectures are built on, and is the primary class you use.
Additionally, X-MLPs relies heavily on factory functions to customize and instantiate the building blocks that make up a particular `XMLP` instance.
Fortunately, this library provides several SOTA MLP blocks out-of-the-box as factory functions.
For example, to implement the ResMLP architecture, you can implement the follow model function:

```python
import haiku as hk
import jax
from einops import rearrange
from x_mlps import XMLP, Affine, resmlp_block_factory

def create_model(patch_size: int, dim: int, depth: int, num_classes: int = 10):
    def model_fn(x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        # Reformat input image into a sequence of patches
        x = rearrange(x, "(h p1) (w p2) c -> (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
        return XMLP(
            num_patches=x.shape[-2],
            dim=dim,
            depth=depth,
            block=resmlp_block_factory,
            # Normalization following the stack of ResMLP blocks
            normalization=lambda num_patches, dim, depth, **kwargs: Affine(dim, **kwargs),
            num_classes=num_classes,
        )(x, is_training=is_training)

    # NOTE: Operating directly on batched data is supported as well.
    return hk.vmap(model_fn, in_axes=(0, None))

model = create_model(patch_size=4, dim=384, depth=12)
model_fn = hk.transform(model)

rng = jax.random.PRNGKey(0)
params = model_fn.init(rng, jnp.ones((1, 32, 32, 3)), False)
```

It's important to note the `XMLP` module _does not_ reformat input data for you (e.g., to a sequence of patches).
As such, you must reformat data manually before feeding it to an `XMLP` module.
The [einops](https://github.com/arogozhnikov/einops) library, which is installed by X-MLPs, provides functions that can help here (e.g., `rearrange`).

**Note**: Like the core Haiku modules, all modules implemented in X-MLPs support batched data and being vectorized via `vmap`.

## X-MLPs Architecture Details

X-MLPs uses a layered approach to construct arbitrary MLP networks. There are three core modules used to create a network's structure:

1. `XSublayer` - bottom level module which wraps arbitrary feedforward functionality.
2. `XBlock` - mid level module consisting of one or more `XSublayer` modules.
3. `XMLP` - top level module which represents a generic MLP network, and is composed of a stack of repeated `XBlock` modules.

To support user-defined modules, each of the above modules support passing arbitrary keyword arguments to child modules.
This is accomplished by prepending arguments with one or more predefined prefixes (including user defined prefixes).
Built-in prefixes include:

1. "block\_" - arguments fed directly to the `XBlock` module.
2. "sublayers\_" - arguments fed to all `XSublayer`s in each `XBlock`.
3. "sublayers{i}\_" - arguments fed to the i-th `XSublayer` in each `XBlock` (where 1 <= i <= # of sublayers).
4. "ff\_" - arguments fed to the feedforward module in a `XSublayer`.

Prefixes must be combined in order when passing them to the `XMLP` module (e.g., "block_sublayer1_ff\_<argument name>").

### XSublayer

The `XSublayer` module is a flexible sublayer wrapper module providing skip connections and pre/post-normalization to an arbitrary child module (specifically, arbitrary feedforward modules e.g., `XChannelFeedForward`).
Child module instances are not passed directly, rather a factory function which creates the child module is instead.
This ensures that individual sublayers can be configured automatically based on depth.

### XBlock

The `XBlock` module is a generic MLP block. It is composed of one or more `XSublayer` modules, passed as factory functions.

### XMLP

At the top level is the `XMLP` module, which represents a generic MLP network.
N `XBlock` modules are stacked together to form a network, created via a common factory function.

## Built-in MLP Architectures

The following architectures have been implemented in the form of `XBlock`s and have corresponding factory functions.

- ResMLP - `resmlp_block_factory`
- MLP-Mixer - `mlpmixer_block_factory`
- gMLP - `gmlp_block_factory`
- S²-MLP - `s2mlp_block_factory`
- FNet - `fft_block_factory`

Importantly, the components that make up these blocks are part of the public API, and thus can be easily mixed and
matched. Additionally, several component variations have been made based on combining ideas from current research.
This includes:

- `fftlinear_block_factory` - an FNet block variation: combines a 1D FFT for patch mixing plus a linear layer.
- `create_shift1d_op` - A 1D shift operation inspired by S²-MLP, making it appropriate for sequence data.
- `create_multishift1d_op` - Like `create_shift1d_op`, but supports multiple shifts of varying sizes.

See their respective docstrings for more information.

## LICENSE

See [LICENSE](LICENSE).

## Citations

```bibtex
@article{Touvron2021ResMLPFN,
  title={ResMLP: Feedforward networks for image classification with data-efficient training},
  author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and Edouard Grave and Gautier Izacard and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Herv'e J'egou},
  journal={ArXiv},
  year={2021},
  volume={abs/2105.03404}
}
```

```bibtex
@article{Tolstikhin2021MLPMixerAA,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Ilya O. Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
  journal={ArXiv},
  year={2021},
  volume={abs/2105.01601}
}
```

```bibtex
@article{Liu2021PayAT,
  title={Pay Attention to MLPs},
  author={Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
  journal={ArXiv},
  year={2021},
  volume={abs/2105.08050}
}
```

```bibtex
@article{Yu2021S2MLPSM,
  title={S2-MLP: Spatial-Shift MLP Architecture for Vision},
  author={Tan Yu and Xu Li and Yunfeng Cai and Mingming Sun and Ping Li},
  journal={ArXiv},
  year={2021},
  volume={abs/2106.07477}
}
```

```bibtex
@article{Touvron2021GoingDW,
  title={Going deeper with Image Transformers},
  author={Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Herv'e J'egou},
  journal={ArXiv},
  year={2021},
  volume={abs/2103.17239}
}
```

```bibtex
@inproceedings{Huang2016DeepNW,
  title={Deep Networks with Stochastic Depth},
  author={Gao Huang and Yu Sun and Zhuang Liu and Daniel Sedra and Kilian Q. Weinberger},
  booktitle={ECCV},
  year={2016}
}
```

```bibtex
@article{LeeThorp2021FNetMT,
  title={FNet: Mixing Tokens with Fourier Transforms},
  author={James P Lee-Thorp and Joshua Ainslie and Ilya Eckstein and Santiago Onta{\~n}{\'o}n},
  journal={ArXiv},
  year={2021},
  volume={abs/2105.03824}
}
```
