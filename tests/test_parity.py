"""Parity checks against the actual Triangle and TetGen wrappers in MeshPy."""

import numpy as np
import pytest

import meshpy.tet as upstream_tet
import meshpy.triangle as upstream_triangle
from mojomeshpy import tet, triangle
from mojomeshpy._lib import lib


def simplex_set(elements):
    return {tuple(sorted(map(int, element))) for element in elements}


def test_triangle_matches_meshpy_without_refinement():
    rng = np.random.default_rng(42)
    points = np.vstack(([[0, 0], [1, 0], [1, 1], [0, 1]], .15 + .7 * rng.random((80, 2))))
    facets = [(0, 1), (1, 2), (2, 3), (3, 0)]
    ours_info = triangle.MeshInfo(); ours_info.set_points(points); ours_info.set_facets(facets)
    their_info = upstream_triangle.MeshInfo(); their_info.set_points(points); their_info.set_facets(facets)
    ours = triangle.build(ours_info, quality_meshing=False, generate_neighbor_lists=True)
    theirs = upstream_triangle.build(their_info, quality_meshing=False)
    assert simplex_set(ours.elements) == simplex_set(theirs.elements)
    assert {tuple(edge) for edge in ours.faces} == {tuple(sorted(edge)) for edge in facets}
    assert ours.neighbors.shape == ours.elements.shape


def test_tet_matches_meshpy_delaunay_mode():
    rng = np.random.default_rng(9)
    outer = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    points = np.vstack((outer, rng.dirichlet([1, 1, 1, 1], size=35)[:, 1:]))
    ours_info = tet.MeshInfo(); ours_info.set_points(points)
    their_info = upstream_tet.MeshInfo(); their_info.set_points(points)
    ours = tet.build(ours_info)
    theirs = upstream_tet.build(their_info, options=upstream_tet.Options("Q"))
    assert simplex_set(ours.elements) == simplex_set(theirs.elements)
    assert ours.neighbors.shape == ours.elements.shape


def test_triangle_boundary_is_checked_not_ignored():
    info = triangle.MeshInfo()
    info.set_points([(0, 0), (1, 0), (0, 1), (.2, .2)])
    info.set_facets([(0, 1), (1, 2), (2, 0)])
    out = triangle.build(info, quality_meshing=False)
    assert simplex_set(out.elements) == {(0, 1, 3), (0, 2, 3), (1, 2, 3)}
    info.set_holes([(0.1, 0.1)])
    with pytest.raises(NotImplementedError, match="holes"):
        triangle.build(info)


def test_tet_boundary_is_checked_not_ignored():
    info = tet.MeshInfo()
    info.set_points([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (.1, .1, .1)])
    info.set_facets([(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)], markers=[4, 5, 6, 7])
    out = tet.build(info)
    assert len(out.elements) == 4
    assert out.facet_markers.tolist() == [4, 5, 6, 7]
    assert out.faces.shape == (4, 3)
    info.set_facets([(0, 1, 2)])
    with pytest.raises(NotImplementedError, match="convex hull"):
        tet.build(info)


def test_simd_copy_tail_preserves_delaunay_parity():
    points2 = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.3, .4]])
    facets2 = [(0, 1), (1, 3), (3, 2), (2, 0)]
    ours_info = triangle.MeshInfo(); ours_info.set_points(points2); ours_info.set_facets(facets2)
    their_info = upstream_triangle.MeshInfo(); their_info.set_points(points2); their_info.set_facets(facets2)
    ours = triangle.build(ours_info, quality_meshing=False)
    theirs = upstream_triangle.build(their_info, quality_meshing=False)
    assert simplex_set(ours.elements) == simplex_set(theirs.elements)

    points3 = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [.1, .2, .1], [.2, .1, .3],
    ])
    ours_info = tet.MeshInfo(); ours_info.set_points(points3)
    their_info = upstream_tet.MeshInfo(); their_info.set_points(points3)
    ours = tet.build(ours_info)
    theirs = upstream_tet.build(their_info, options=upstream_tet.Options("Q"))
    assert simplex_set(ours.elements) == simplex_set(theirs.elements)


def test_meshpy_style_helpers_and_markers():
    info = triangle.MeshInfo()
    info.set_points([(0, 0), (1, 0), (0, 1)], point_markers=[3, 4, 5])
    info.set_facets([(0, 1), (1, 2), (2, 0)], facet_markers=[7, 8, 9])
    out = triangle.build(info, quality_meshing=False)
    assert out.point_markers.tolist() == [3, 4, 5]
    assert out.facet_markers.tolist() == [7, 8, 9]
    points, facets, markers = triangle.subdivide_facets(2, [(0, 0), (1, 0)], [(0, 1)], [6])
    assert points[2] == (0.5, 0.0) and facets == [(0, 2), (2, 1)] and markers == [6, 6]


@pytest.mark.parametrize("module, dimension", [(triangle, 2), (tet, 3)])
def test_documented_meshinfo_helpers(module, dimension):
    info = module.MeshInfo()
    points = np.eye(dimension + 1, dimension)
    info.set_points(points)
    info.set_holes([])
    info.set_regions([])
    copied = info.copy()
    copied.points[0, 0] = 99
    assert info.holes.shape == (0, dimension)
    assert info.regions.shape == (0, dimension + 2)
    assert info.points[0, 0] != copied.points[0, 0]


def test_integer_inputs_are_not_silently_truncated():
    info = triangle.MeshInfo()
    with pytest.raises(TypeError, match="integers"):
        info.set_points([(0, 0), (1, 0), (0, 1)], point_markers=[1.5, 2, 3])
    with pytest.raises(TypeError, match="integers"):
        info.set_facets([(0.5, 1)])
    with pytest.raises(ValueError, match="exactly representable"):
        info.set_points([[-(2**53 + 1), 0], [1, 0], [0, 1]])


def test_unsupported_options_fail_instead_of_being_ignored():
    info = triangle.MeshInfo(); info.set_points([(0, 0), (1, 0), (0, 1)])
    with pytest.raises(NotImplementedError, match="quality"):
        triangle.build(info)
    tet_info = tet.MeshInfo(); tet_info.set_points([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    assert len(tet.build(tet_info, options=tet.Options("Q")).elements) == 1
    with pytest.raises(NotImplementedError, match="only tet.Options"):
        tet.build(tet_info, options=tet.Options("pq"))


def test_ffi_rejects_null_addresses_before_pointer_reconstruction():
    assert lib().mmp_delaunay2d(0, 3, 0, 4, 0, 0, 0, 0) == -2
    assert lib().mmp_delaunay3d(0, 4, 0, 8, 0, 0, 0, 0) == -2


@pytest.mark.parametrize("module, dimension", [(triangle, 2), (tet, 3)])
def test_point_shape_is_validated(module, dimension):
    info = module.MeshInfo()
    with pytest.raises(ValueError):
        info.set_points(np.zeros((3, dimension + 1)))
