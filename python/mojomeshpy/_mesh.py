"""Small NumPy-backed result object shared by the 2-D and 3-D APIs."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ._lib import f64, i64


class MeshInfoBase:
    """The MeshPy array attributes used by the supported generation subset."""

    dimension: int

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.points = np.empty((0, dimension), dtype=np.float64)
        self.point_markers = np.empty(0, dtype=np.int64)
        self.facets = []
        self.facet_markers = np.empty(0, dtype=np.int64)
        self.holes = np.empty((0, dimension), dtype=np.float64)
        self.regions = np.empty((0, dimension + 2), dtype=np.float64)
        self.elements = np.empty((0, dimension + 1), dtype=np.int64)
        self.element_attributes = np.empty((0, 0), dtype=np.float64)
        self.element_volumes = np.empty(0, dtype=np.float64)
        self.faces = np.empty((0, dimension), dtype=np.int64)
        self.face_markers = np.empty(0, dtype=np.int64)
        self.neighbors = np.empty((0, dimension + 1), dtype=np.int64)
        self.number_of_point_attributes = 0
        self.number_of_element_attributes = 0

    def set_points(self, points, point_markers=None):
        arr = f64(points, self.dimension)
        self.points = arr
        if point_markers is None:
            self.point_markers = np.zeros(len(arr), dtype=np.int64)
        else:
            self.point_markers = i64(point_markers, name="point_markers", shape=(len(arr),))

    def set_holes(self, hole_starts):
        holes = np.ascontiguousarray(hole_starts, dtype=np.float64)
        if holes.size == 0:
            holes = np.empty((0, self.dimension), dtype=np.float64)
        if holes.ndim != 2 or holes.shape[1] != self.dimension:
            raise ValueError(f"holes must have shape (n, {self.dimension})")
        if not np.isfinite(holes).all():
            raise ValueError("holes must be finite")
        self.holes = holes

    def set_regions(self, regions):
        arr = np.ascontiguousarray(regions, dtype=np.float64)
        if arr.size == 0:
            arr = np.empty((0, self.dimension + 2), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < self.dimension:
            raise ValueError("regions must start with one coordinate per dimension")
        if not np.isfinite(arr).all():
            raise ValueError("regions must be finite")
        self.regions = arr

    def copy(self):
        return deepcopy(self)

    def dump(self):
        print(f"{len(self.points)} points, {len(self.elements)} elements")


def boundary_faces(elements: np.ndarray, vertices_per_face: int) -> np.ndarray:
    counts: dict[tuple[int, ...], int] = {}
    for element in elements:
        for omit in range(len(element)):
            face = tuple(sorted(np.delete(element, omit).tolist()))
            counts[face] = counts.get(face, 0) + 1
    return np.asarray([face for face, count in counts.items() if count == 1], dtype=np.int64).reshape(-1, vertices_per_face)


def neighbor_table(elements: np.ndarray) -> np.ndarray:
    result = np.full((len(elements), elements.shape[1]), -1, dtype=np.int64)
    owner: dict[tuple[int, ...], tuple[int, int]] = {}
    for i, element in enumerate(elements):
        for omit in range(len(element)):
            key = tuple(sorted(np.delete(element, omit).tolist()))
            if key in owner:
                j, other_omit = owner[key]
                result[i, omit] = j
                result[j, other_omit] = i
            else:
                owner[key] = (i, omit)
    return result
