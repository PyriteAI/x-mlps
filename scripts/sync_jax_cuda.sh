#!/bin/sh

poetry run pip install --force-reinstall --no-deps jaxlib==0.3.10+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html
