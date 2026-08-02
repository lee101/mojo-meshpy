"""Measure the identical no-refinement Delaunay workloads in both libraries."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import meshpy.tet as upstream_tet
import meshpy.triangle as upstream_triangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from mojomeshpy import tet, triangle


def elapsed(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter(); fn(); samples.append(time.perf_counter() - start)
    return min(samples) * 1e3


def triangle_case(module, points, facets):
    info = module.MeshInfo(); info.set_points(points); info.set_facets(facets)
    if module is triangle:
        return lambda: module.build(info, quality_meshing=False)
    return lambda: module.build(info, quality_meshing=False)


def tet_case(module, points):
    info = module.MeshInfo(); info.set_points(points)
    if module is tet:
        return lambda: module.build(info)
    return lambda: module.build(info, options=module.Options("Q"))


def main():
    rng = np.random.default_rng(2026)
    points2 = np.vstack(([[0, 0], [1, 0], [1, 1], [0, 1]], .2 + .6 * rng.random((196, 2))))
    points3 = np.vstack(([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                          rng.dirichlet([1, 1, 1, 1], size=56)[:, 1:]))
    rows = [
        ("2-D Delaunay, 200 points", elapsed(triangle_case(triangle, points2, [(0, 1), (1, 2), (2, 3), (3, 0)]), 5),
         elapsed(triangle_case(upstream_triangle, points2, [(0, 1), (1, 2), (2, 3), (3, 0)]), 5)),
        ("3-D Delaunay, 60 points", elapsed(tet_case(tet, points3), 3), elapsed(tet_case(upstream_tet, points3), 3)),
    ]
    print(f"Machine: {platform.platform()} | Python {platform.python_version()}")
    print("| kernel | Mojo (ms) | MeshPy (ms) | Mojo/MeshPy |")
    print("|---|---:|---:|---:|")
    for name, mojo_ms, meshpy_ms in rows:
        print(f"| {name} | {mojo_ms:.3f} | {meshpy_ms:.3f} | {mojo_ms / meshpy_ms:.2f}x |")


if __name__ == "__main__":
    main()
