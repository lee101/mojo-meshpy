"""ctypes bridge for the Mojo Delaunay kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(os.environ.get("MOJOMESHPY_LIB", ROOT / "dist" / "libmojo-meshpy.so"))
I = ctypes.c_int64


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> Path:
    sources = list((ROOT / "src").glob("*.mojo"))
    if not force and LIB.exists() and all(LIB.stat().st_mtime >= p.stat().st_mtime for p in sources):
        return LIB
    proc = subprocess.run(["bash", str(ROOT / "build" / "build.sh")], cwd=ROOT,
                          text=True, capture_output=True, timeout=1800)
    if proc.returncode or not LIB.exists():
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_handle: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        _handle = ctypes.CDLL(str(build()))
        _handle.mmp_delaunay2d.argtypes = [I] * 8
        _handle.mmp_delaunay2d.restype = I
        _handle.mmp_delaunay3d.argtypes = [I] * 8
        _handle.mmp_delaunay3d.restype = I
    return _handle


def f64(a, columns: int) -> np.ndarray:
    raw = np.asarray(a)
    if raw.dtype.kind not in "iuf":
        raise TypeError("points must be a real numeric array")
    if raw.dtype.kind in "iu" and (np.any(raw > 2**53) or np.any(raw < -2**53)):
        raise ValueError("integer coordinates must be exactly representable as float64")
    arr = np.ascontiguousarray(raw, dtype=np.float64)
    if raw.dtype.kind == "f" and raw.dtype.itemsize > arr.dtype.itemsize:
        if not np.array_equal(arr.astype(raw.dtype), raw):
            raise ValueError("coordinates must be exactly representable as float64")
    if arr.ndim != 2 or arr.shape[1] != columns:
        raise ValueError(f"points must have shape (n, {columns})")
    if not np.isfinite(arr).all():
        raise ValueError("points must be finite")
    return arr


def i64(a, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Return an owned, contiguous int64 array without truncating user values."""
    raw = np.asarray(a)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    info = np.iinfo(np.int64)
    if raw.dtype.kind == "u" and np.any(raw > info.max):
        raise OverflowError(f"{name} values do not fit in int64")
    arr = np.ascontiguousarray(raw, dtype=np.int64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return arr


def addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)
