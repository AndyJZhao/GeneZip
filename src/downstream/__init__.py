"""Downstream task modules used by release scripts."""

from typing import Any


def load_data(*args: Any, **kwargs: Any):
    from .dlb_data import load_data as _load_data

    return _load_data(*args, **kwargs)

__all__ = ["load_data"]
