"""MeshPy-compatible 2-D point-set Delaunay triangulation."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib
from ._mesh import MeshInfoBase, boundary_faces, topology


class MeshInfo(MeshInfoBase):
    def __init__(self):
        super().__init__(2)

    def set_facets(self, facets, facet_markers=None):
        facets = np.asarray(facets)
        if facets.size == 0:
            facets = np.empty((0, 2), dtype=np.int64)
        facets = i64(facets, name="facets")
        if facets.ndim != 2 or facets.shape[1] != 2:
            raise ValueError("triangle facets must be pairs of point indices")
        self.facets = [tuple(map(int, facet)) for facet in facets]
        self.facet_markers = (np.zeros(len(facets), dtype=np.int64) if facet_markers is None
                              else i64(facet_markers, name="facet_markers", shape=(len(facets),)))


def _check_supported(mesh_info: MeshInfo, max_volume, refinement_func, quality_meshing,
                     min_angle, attributes, volume_constraints, mesh_order):
    if len(mesh_info.holes) or len(mesh_info.regions):
        raise NotImplementedError("holes and regions need constrained Triangle meshing")
    if max_volume is not None or refinement_func is not None:
        raise NotImplementedError("Steiner-point refinement is outside the Delaunay subset")
    if quality_meshing or min_angle is not None or volume_constraints:
        raise NotImplementedError("quality and volume meshing are outside the Delaunay subset")
    if attributes or mesh_order not in (None, 1):
        raise NotImplementedError("element attributes and higher-order elements are unsupported")
    if mesh_info.facets:
        # Delaunay respects a convex hull. A PLC with a concavity or an internal
        # segment must be handled by a constrained kernel, never silently ignored.
        degrees = np.zeros(len(mesh_info.points), dtype=np.int64)
        for a, b in mesh_info.facets:
            if not (0 <= a < len(degrees) and 0 <= b < len(degrees)):
                raise ValueError("facet index out of bounds")
            degrees[a] += 1; degrees[b] += 1
        if np.any((degrees != 0) & (degrees != 2)):
            raise NotImplementedError("only a single convex boundary loop is supported")


def build(mesh_info, verbose=False, refinement_func=None, attributes=False,
          volume_constraints=False, max_volume=None, allow_boundary_steiner=True,
          allow_volume_steiner=True, quality_meshing=True, generate_edges=None,
          generate_faces=False, min_angle=None, mesh_order=None,
          generate_neighbor_lists=False):
    _check_supported(mesh_info, max_volume, refinement_func, quality_meshing, min_angle,
                     attributes, volume_constraints, mesh_order)
    points = f64(mesh_info.points, 2)
    n = len(points)
    cap = max(32, 8 * n + 64)
    result = np.empty((cap, 3), dtype=np.int64)
    work_points = np.empty((n + 4, 2), dtype=np.float64)
    work_triangles = np.empty((cap, 3), dtype=np.int64)
    edges = np.empty((3 * cap, 2), dtype=np.int64)
    count = lib().mmp_delaunay2d(addr(points), n, addr(result), cap, addr(work_points),
                                  addr(work_triangles), addr(edges))
    if count == -2:
        raise RuntimeError("Mojo kernel rejected a null buffer address")
    if count < 0:
        raise RuntimeError("Delaunay scratch capacity exhausted")
    out = MeshInfo()
    out.set_points(points, mesh_info.point_markers)
    out.facets = list(mesh_info.facets)
    out.facet_markers = mesh_info.facet_markers.copy()
    out.elements = result[:count]
    if generate_neighbor_lists:
        out.faces, out.neighbors = topology(out.elements)
    else:
        out.faces = boundary_faces(out.elements, 2)
    out.face_markers = np.zeros(len(out.faces), dtype=np.int64)
    if mesh_info.facets:
        actual = {tuple(face) for face in out.faces}
        expected = {tuple(sorted(face)) for face in mesh_info.facets}
        if actual != expected:
            raise NotImplementedError("facets must be the point set's convex hull")
    return out


def refine(input_p, verbose=False, refinement_func=None, quality_meshing=True,
           min_angle=None, generate_neighbor_lists=False):
    raise NotImplementedError("refine needs Triangle's constrained refinement kernel")


def subdivide_facets(subdivisions, points, facets, facet_markers=None):
    new_points = [tuple(point) for point in points]
    new_facets, new_markers = [], []
    for i, (a, b) in enumerate(facets):
        steps = int(subdivisions[i] if np.ndim(subdivisions) else subdivisions)
        if steps < 1:
            raise ValueError("subdivisions must be positive")
        chain = [a]
        pa, pb = np.asarray(points[a], float), np.asarray(points[b], float)
        for j in range(1, steps):
            chain.append(len(new_points)); new_points.append(tuple(pa + (pb - pa) * j / steps))
        chain.append(b)
        marker = 0 if facet_markers is None else facet_markers[i]
        for left, right in zip(chain, chain[1:]):
            new_facets.append((left, right)); new_markers.append(marker)
    return new_points, new_facets, new_markers
