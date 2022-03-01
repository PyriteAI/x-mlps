# Radio-Frequency Machine Learning Examples

These examples demonstrate using X-MLPs for automated modulation recognition, a class a problem under the domain of
radio-frequency machine learning. In contrast to the vision examples, which demonstrate a simple setup that can be run
on most consumer GPUs, these examples require fairly powerful hardware. For reference, these scripts were developed on
an AWS g5.12xlarge instance, which has 4 A10G GPUs, 48 CPU cores, and 192 GiB of RAM.

## Getting Started

These examples use the RadioML 2018.01A. Unfortunately, it looks like [DeepSig](https://www.deepsig.ai/?hsLang=en)
removed public access to this dataset (as well as their others) sometime during development. It may be possible to
reach out to the DeepSig team and request access.

If you do have access to the dataset, use the `split_radioml.py` script to generate a train/validation/test split.
Once that is done, you can run the examples in here. Note that these examples assume the splits are placed in a `data/`
folder in this directory.
