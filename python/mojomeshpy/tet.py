"""MeshPy-compatible 3-D point-set Delaunay tetrahedralization."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib
from ._mesh import MeshInfoBase, boundary_faces, neighbor_table


class MeshInfo(MeshInfoBase):
    def __init__(self):
        super().__init__(3)

    def set_facets(self, facets, markers=None):
        values = np.asarray(facets)
        if values.size == 0:
            values = np.empty((0, 3), dtype=np.int64)
        values = i64(values, name="facets")
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("tet facets must be triples of point indices")
        self.facets = [tuple(map(int, facet)) for facet in values]
        self.facet_markers = (np.zeros(len(self.facets), dtype=np.int64) if markers is None
                              else i64(markers, name="markers", shape=(len(self.facets),)))


class Options:
    """TetGen-compatible point-Delaunay option container; only ``Q`` is accepted."""
    def __init__(self, switches, **kwargs):
        self.switches = switches
        for name, value in kwargs.items():
            setattr(self, name, value)


def build(mesh_info, options=None, verbose=False, attributes=False,
          volume_constraints=False, max_volume=None, diagnose=False,
          insert_points=None):
    if len(mesh_info.holes) or len(mesh_info.regions):
        raise NotImplementedError("PLC holes and regions need the TetGen kernel")
    if max_volume is not None or insert_points is not None:
        raise NotImplementedError("Steiner-point insertion is outside the Delaunay subset")
    if attributes or volume_constraints:
        raise NotImplementedError("element attributes and volume constraints are unsupported")
    if options is not None and (not isinstance(options, Options) or options.switches != "Q" or vars(options) != {"switches": "Q"}):
        raise NotImplementedError("only tet.Options('Q') is supported")
    points = f64(mesh_info.points, 3)
    n = len(points)
    cap = max(128, 32 * n + 256)
    result = np.empty((cap, 4), dtype=np.int64)
    work_points = np.empty((n + 4, 3), dtype=np.float64)
    work_tets = np.empty((cap, 4), dtype=np.int64)
    live = np.empty(cap, dtype=np.int64)
    faces = np.empty((4 * cap, 3), dtype=np.int64)
    count = lib().mmp_delaunay3d(addr(points), n, addr(result), cap, addr(work_points),
                                  addr(work_tets), addr(live), addr(faces))
    if count == -2:
        raise RuntimeError("Mojo kernel rejected a null buffer address")
    if count < 0:
        raise RuntimeError("tetrahedralization scratch capacity exhausted")
    out = MeshInfo()
    out.set_points(points, mesh_info.point_markers)
    out.facets = list(mesh_info.facets)
    out.facet_markers = mesh_info.facet_markers.copy()
    out.elements = result[:count].copy()
    out.faces = boundary_faces(out.elements, 3) if count else np.empty((0, 3), dtype=np.int64)
    out.face_markers = np.zeros(len(out.faces), dtype=np.int64)
    if mesh_info.facets:
        expected = {tuple(sorted(face)) for face in mesh_info.facets}
        actual = {tuple(face) for face in out.faces}
        if actual != expected:
            raise NotImplementedError("facets must be the point set's convex hull")
    out.neighbors = neighbor_table(out.elements)
    return out
