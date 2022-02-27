import numpy as np
import tables
import torch
from torch.utils.data import Dataset, Subset, TensorDataset


class RadioML2018(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        super().__init__()

        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self._file = None

    @property
    def labels(self):
        self.open()
        return np.array(self._file.root.Y)

    @property
    def snr(self):
        self.open()
        return np.array(self._file.root.Z)

    def __getitem__(self, index):
        self.open()

        x, y, z = self._file.root.X[index], self._file.root.Y[index], self._file.root.Z[index]
        x, y, z = torch.from_numpy(x), torch.as_tensor(y), torch.as_tensor(z)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y, z

    def __len__(self):
        try:
            return len(self._file.root.X)
        except AttributeError:
            self.open()
            length = len(self._file.root.X)
            self.close()

            return length

    def open(self):
        if self._file is None:
            self._file = tables.open_file(self.path)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def create_subset_indices(filepath, min_snr=None, max_snr=None):
    with tables.open_file(filepath) as f:
        snrs = np.array(f.root.Z).squeeze()
        if min_snr is None:
            min_snr = np.iinfo(snrs.dtype).min
        if max_snr is None:
            max_snr = np.iinfo(snrs.dtype).max
        return np.argwhere((snrs >= min_snr) & (snrs <= max_snr)).squeeze()


def load_dataset(path, min_snr=None, max_snr=None, in_memory=False):
    if in_memory:
        with tables.open_file(path) as f:
            x, y, z = np.array(f.root.X), np.array(f.root.Y), np.array(f.root.Z)
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z))
    else:
        ds = RadioML2018(path)
    if min_snr is not None:
        indices = create_subset_indices(path, min_snr, max_snr=max_snr)
        ds = Subset(ds, indices=indices.tolist())
    return ds


__all__ = ["RadioML2018", "load_dataset"]
