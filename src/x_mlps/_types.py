from typing import Any, Optional, Protocol

import haiku as hk


class XModuleFactory(Protocol):
    """Defines a common factory function interface for all X-MLP modules."""

    def __call__(self, num_patches: int, dim: int, depth: int, name: Optional[str] = None, **kwargs: Any) -> hk.Module:
        ...


__all__ = ["XModuleFactory"]
