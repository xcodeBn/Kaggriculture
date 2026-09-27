"""Clean environment loader for Kaggriculture.

Silences third-party arm64 architecture warnings (libcg.dylib) and missing
optional module warnings (pyspiel) emitted by unrelated kaggle-environments envs.
"""
from __future__ import annotations

import os
from typing import Any, Callable

_cached_make: Callable[..., Any] | None = None


def get_clean_make() -> Callable[..., Any]:
    global _cached_make
    if _cached_make is not None:
        return _cached_make

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        from kaggle_environments import make
        _cached_make = make
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)

    return _cached_make
