# mojo-meshpy

`mojo-meshpy` is a standalone Mojo implementation of the point-set Delaunay
part of [MeshPy](https://documen.tician.de/meshpy/): 2-D triangle generation
and 3-D tetrahedralization.  It exposes NumPy-backed Python objects with the
same `triangle.MeshInfo`, `tet.MeshInfo`, and `build` entry-point names as the
covered MeshPy subset.

This is deliberately not a wrapper around Triangle or TetGen.  The incremental
Bowyer-Watson kernels are compiled from one Mojo translation unit and called by
`ctypes` through a narrow raw-buffer ABI.

## Coverage

Covered and tested:

- `mojomeshpy.triangle.MeshInfo`, `set_points`, `set_facets`, `set_holes`,
  `set_regions`, `copy`, `triangle.build`, and `subdivide_facets`.
- `mojomeshpy.tet.MeshInfo`, `set_points`, `set_facets`, `set_holes`,
  `set_regions`, `copy`, `tet.Options("Q")`, and `tet.build`.
- Unconstrained 2-D and 3-D Delaunay meshes.  A supplied boundary is accepted
  when it is exactly the resulting convex hull; it is verified after meshing.
- Point/facet markers, output faces, and element-neighbor lists.

Not covered: constrained segments/facets, holes, regions, quality or
volume-driven Steiner insertion, higher-order elements, element attributes,
and `triangle.refine`. Only `quality_meshing=False` is accepted in
`triangle.build`; other unsupported requests raise `NotImplementedError`
instead of returning a mesh that silently violates a PLC constraint.

## Install and run

```bash
pixi install
pixi run build
pixi run test
```

The package is source-layout based while developing; Pixi sets `PYTHONPATH` for
all tasks.  This complete example runs from the checkout:

```bash
pixi run python - <<'PY'
from mojomeshpy import triangle

info = triangle.MeshInfo()
info.set_points([(0, 0), (1, 0), (1, 1), (0, 1), (.4, .6)])
info.set_facets([(0, 1), (1, 2), (2, 3), (3, 0)])
mesh = triangle.build(info, quality_meshing=False, generate_neighbor_lists=True)
print(mesh.elements.tolist())
PY
```

## How it works

`src/capi.mojo` uses incremental Bowyer-Watson insertion. Coordinates are
copied into a row-major `float64` work array (`n x 2` or `n x 3`); simplices
and cavity scratch space are NumPy-owned row-major `int64` arrays. The C ABI takes
their addresses as `Int`, rejects null addresses before reconstructing typed
Mojo pointers, and returns a simplex count. The Python layer validates shapes,
finiteness, contiguity, and integer ranges, keeps every NumPy buffer alive for
the duration of the call, and exposes a trimmed view of the completed result.

Predicates use `float64`, so this is intended for non-degenerate point sets.
The tests compare its simplices against Triangle with `quality_meshing=False`
and TetGen's `Q` mode on deterministic point clouds.

## Benchmarks

Measured by `pixi run bench` on this checkout. Each value is the fastest of
repeated identical runs; both implementations receive the same input, and
upstream uses no-refinement modes.

| kernel | Mojo (ms) | MeshPy (ms) | Mojo/MeshPy |
|---|---:|---:|---:|
| 2-D Delaunay, 200 points | 0.738 | 0.128 | 5.77x |
| 3-D Delaunay, 60 points | 0.704 | 0.235 | 3.00x |

Profiling showed that repeated Python `np.delete` calls in boundary and neighbor
reconstruction dominated both workloads. Topology reconstruction now builds one
contiguous face matrix, sorts it once, and groups adjacent equal faces. The
native insertion kernel compacts dead simplices on every insertion so later
predicates do not revisit them, and the result stays in its NumPy-owned FFI
buffer without a final copy.

The CPU copies input coordinates with the target's native Float64 SIMD width and
a scalar tail. The more arithmetic-heavy 3-D in-sphere scan evaluates active
tetrahedra in SIMD batches, again with a scalar tail. Incremental cavity updates
have loop-carried topology dependencies and small active frontiers, so thread
launch and merge overhead outweigh useful parallel work; no threaded path was
added. No GPU path is included: predicates gather irregular vertex data and
perform less than roughly two floating-point operations per byte moved, so the
work cannot justify device allocation and transfers.
