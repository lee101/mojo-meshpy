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


def _sorted_faces(elements: np.ndarray):
    width = elements.shape[1]
    if width == 3:
        faces = np.empty((len(elements) * 3, 2), dtype=np.int64)
        faces[0::3] = elements[:, (1, 2)]
        faces[1::3] = elements[:, (0, 2)]
        faces[2::3] = elements[:, (0, 1)]
    else:
        columns = np.asarray([
            [column for column in range(width) if column != omit]
            for omit in range(width)
        ])
        faces = elements[:, columns].reshape(-1, width - 1)
    faces.sort(axis=1)
    if width == 3:
        base = int(faces.max()) + 1
        order = np.argsort(faces[:, 0] * base + faces[:, 1])
    else:
        order = np.lexsort(tuple(faces[:, column] for column in range(width - 2, -1, -1)))
    sorted_faces = faces[order]
    same = np.all(sorted_faces[1:] == sorted_faces[:-1], axis=1)
    internal = np.zeros(len(faces), dtype=bool)
    internal[:-1] |= same
    internal[1:] |= same
    return sorted_faces[~internal], order, same


def topology(elements: np.ndarray):
    if not len(elements):
        width = elements.shape[1]
        return (np.empty((0, width - 1), dtype=np.int64),
                np.empty((0, width), dtype=np.int64))
    boundary, order, paired = _sorted_faces(elements)
    width = elements.shape[1]
    neighbors = np.full((len(elements), width), -1, dtype=np.int64)
    left = order[:-1][paired]
    right = order[1:][paired]
    left_element, left_omit = np.divmod(left, width)
    right_element, right_omit = np.divmod(right, width)
    neighbors[left_element, left_omit] = right_element
    neighbors[right_element, right_omit] = left_element
    return boundary, neighbors


def boundary_faces(elements: np.ndarray, vertices_per_face: int) -> np.ndarray:
    if not len(elements):
        return np.empty((0, vertices_per_face), dtype=np.int64)
    return _sorted_faces(elements)[0]


def neighbor_table(elements: np.ndarray) -> np.ndarray:
    return topology(elements)[1]
