"""Loss registry for dynamic lookup."""

from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    """Register a loss class under the given name."""
    _REGISTRY[name] = cls


def get(name: str) -> type:
    """Look up a registered loss class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_losses() -> list[str]:
    """Return sorted list of registered loss names."""
    return sorted(_REGISTRY)


def create(name: str, **kwargs: Any) -> Any:
    """Create a loss instance by name."""
    cls = get(name)
    return cls(**kwargs)
