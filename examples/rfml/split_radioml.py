#!/usr/bin/env python3

from pathlib import Path
from typing import Sequence

import numpy as np
import tables
import typer

VALID_RATIO = 1 / 16
TEST_RATIO = 1 / 8
TRAIN_RATIO = 1 - (VALID_RATIO + TEST_RATIO)


def create_h5(f: tables.File, x_shape: Sequence[int], y_shape: Sequence[int], z_shape: Sequence[int]):
    x = f.create_carray(
        f.root,
        "X",
        atom=tables.ComplexAtom(itemsize=8),
        shape=x_shape,
        filters=tables.Filters(complevel=9, complib="blosc"),
    )
    y = f.create_carray(
        f.root,
        "Y",
        atom=tables.Int64Atom(),
        shape=y_shape,
        filters=tables.Filters(complevel=9, complib="blosc"),
    )
    z = f.create_carray(
        f.root,
        "Z",
        atom=tables.Int64Atom(),
        shape=z_shape,
        filters=tables.Filters(complevel=9, complib="blosc"),
    )

    return x, y, z


def main(
    input: Path,
    output_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
):
    np.random.seed(seed)

    train = output_dir / "train.h5"
    valid = output_dir / "valid.h5"
    test = output_dir / "test.h5"

    with tables.open_file(input) as infile, tables.open_file(train, mode="w") as trf, tables.open_file(
        valid, mode="w"
    ) as vf, tables.open_file(test, mode="w") as tef:
        valid_size = int(len(infile.root.X) * valid_ratio)
        assert (  # nosec
            valid_size == len(infile.root.X) * valid_ratio
        ), "number of examples not divisible by given 'valid_ratio'"
        test_size = int(len(infile.root.X) * test_ratio)
        assert test_size == len(infile.root.X) * test_ratio, "number of examples not divisible by 'test_ratio'"  # nosec
        train_size = len(infile.root.X) - valid_size - test_size

        trf_x, trf_y, trf_z = create_h5(trf, (train_size, infile.root.X.shape[1]), (train_size,), (train_size,))
        vf_x, vf_y, vf_z = create_h5(vf, (valid_size, infile.root.X.shape[1]), (valid_size,), (valid_size,))
        tef_x, tef_y, tef_z = create_h5(tef, (test_size, infile.root.X.shape[1]), (test_size,), (test_size,))

        y = np.argmax(infile.root.Y, axis=1)
        z = np.squeeze(infile.root.Z)

        train_count, valid_count, test_count = 0, 0, 0
        for y_val in np.unique(y):
            for z_val in np.unique(z):
                indices = np.argwhere((y == y_val) & (z == z_val)).squeeze()

                n_train = int(len(indices) * train_ratio)
                assert n_train == len(indices) * train_ratio, "number of examples not evenly divisible"  # nosec
                n_valid = int(len(indices) * valid_ratio)
                assert n_valid == len(indices) * valid_ratio, "number of examples not evenly divisible"  # nosec

                x_data, y_data, z_data = (
                    infile.root.X[indices, ...],
                    infile.root.Y[indices, ...],
                    infile.root.Z[indices, ...],
                )

                permutation = np.random.permutation(len(x_data))
                x_data, y_data, z_data = x_data[permutation], y_data[permutation], z_data[permutation]
                x_data = x_data[..., 0] + 1j * x_data[..., 1]
                y_data = y_data.argmax(axis=1)
                z_data = z_data.squeeze()

                end_train_idx = n_train
                start_valid_idx, end_valid_idx = end_train_idx, end_train_idx + n_valid
                start_test_idx = end_valid_idx
                x_train, x_valid, x_test = (
                    x_data[:end_train_idx],
                    x_data[start_valid_idx:end_valid_idx],
                    x_data[start_test_idx:],
                )
                y_train, y_valid, y_test = (
                    y_data[:end_train_idx],
                    y_data[start_valid_idx:end_valid_idx],
                    y_data[start_test_idx:],
                )
                z_train, z_valid, z_test = (
                    z_data[:end_train_idx],
                    z_data[start_valid_idx:end_valid_idx],
                    z_data[start_test_idx:],
                )

                start, end = train_count, train_count + len(x_train)
                trf_x[start:end] = x_train
                trf_y[start:end] = y_train
                trf_z[start:end] = z_train
                train_count = end

                start, end = valid_count, valid_count + len(x_valid)
                vf_x[start:end] = x_valid
                vf_y[start:end] = y_valid
                vf_z[start:end] = z_valid
                valid_count = end

                start, end = test_count, test_count + len(x_test)
                tef_x[start:end] = x_test
                tef_y[start:end] = y_test
                tef_z[start:end] = z_test
                test_count = end


if __name__ == "__main__":
    typer.run(main)
