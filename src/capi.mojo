"""Incremental Delaunay kernels exposed through a small C ABI.

The caller supplies all scratch buffers.  This keeps ownership in NumPy and
lets the exported functions remain non-parametric in the Mojo C ABI.
"""

from std.math import iota
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()


def orient2(p: FPtr, a: Int, b: Int, c: Int) -> Float64:
    return (p[b * 2] - p[a * 2]) * (p[c * 2 + 1] - p[a * 2 + 1]) - \
           (p[b * 2 + 1] - p[a * 2 + 1]) * (p[c * 2] - p[a * 2])


def in_circle(p: FPtr, a: Int, b: Int, c: Int, q: Int) -> Bool:
    var qx = p[q * 2]
    var qy = p[q * 2 + 1]
    var pax = p[a * 2]
    var pay = p[a * 2 + 1]
    var pbx = p[b * 2]
    var pby = p[b * 2 + 1]
    var pcx = p[c * 2]
    var pcy = p[c * 2 + 1]
    var ax = pax - qx
    var ay = pay - qy
    var bx = pbx - qx
    var by = pby - qy
    var cx = pcx - qx
    var cy = pcy - qy
    var det = (ax * ax + ay * ay) * (bx * cy - by * cx) - \
              (bx * bx + by * by) * (ax * cy - ay * cx) + \
              (cx * cx + cy * cy) * (ax * by - ay * bx)
    return det * ((pbx - pax) * (pcy - pay) - (pby - pay) * (pcx - pax)) > 0.0


@export("mmp_delaunay2d")
def mmp_delaunay2d(
    points_addr: Int, n: Int, result_addr: Int, cap: Int, work_points_addr: Int,
    work_triangles_addr: Int, edges_addr: Int
) abi("C") -> Int:
    if n < 3 or cap < 4:
        return 0
    if points_addr == 0 or result_addr == 0 or work_points_addr == 0 or \
            work_triangles_addr == 0 or edges_addr == 0:
        return -2
    var points = FPtr(unsafe_from_address=points_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    var p = FPtr(unsafe_from_address=work_points_addr)
    var triangles = IPtr(unsafe_from_address=work_triangles_addr)
    var edges = IPtr(unsafe_from_address=edges_addr)
    var xmin = points[0]
    var xmax = xmin
    var ymin = points[1]
    var ymax = ymin
    var point_values = n * 2
    var offset = 0
    while offset + W <= point_values:
        p.store(offset, points.load[width=W](offset))
        offset += W
    while offset < point_values:
        p[offset] = points[offset]
        offset += 1
    for i in range(n):
        if points[i * 2] < xmin:
            xmin = points[i * 2]
        if points[i * 2] > xmax:
            xmax = points[i * 2]
        if points[i * 2 + 1] < ymin:
            ymin = points[i * 2 + 1]
        if points[i * 2 + 1] > ymax:
            ymax = points[i * 2 + 1]
    var span = xmax - xmin
    if ymax - ymin > span:
        span = ymax - ymin
    if span <= 0.0:
        return 0
    var mx = (xmin + xmax) * 0.5
    var my = (ymin + ymax) * 0.5
    p[n * 2] = mx - 8.0 * span
    p[n * 2 + 1] = my - 8.0 * span
    p[(n + 1) * 2] = mx + 8.0 * span
    p[(n + 1) * 2 + 1] = my - 8.0 * span
    p[(n + 2) * 2] = mx + 8.0 * span
    p[(n + 2) * 2 + 1] = my + 8.0 * span
    p[(n + 3) * 2] = mx - 8.0 * span
    p[(n + 3) * 2 + 1] = my + 8.0 * span
    triangles[0] = n
    triangles[1] = n + 1
    triangles[2] = n + 2
    triangles[3] = n
    triangles[4] = n + 2
    triangles[5] = n + 3
    var nt = 2
    for q in range(n):
        var ne = 0
        var write = 0
        var t = 0
        while t < nt:
            var a = triangles[t * 3]
            var b = triangles[t * 3 + 1]
            var c = triangles[t * 3 + 2]
            if in_circle(p, a, b, c, q):
                edges[ne * 2] = a
                edges[ne * 2 + 1] = b
                ne += 1
                edges[ne * 2] = b
                edges[ne * 2 + 1] = c
                ne += 1
                edges[ne * 2] = c
                edges[ne * 2 + 1] = a
                ne += 1
            else:
                if write != t:
                    triangles[write * 3] = a
                    triangles[write * 3 + 1] = b
                    triangles[write * 3 + 2] = c
                write += 1
            t += 1
        nt = write
        for e in range(ne):
            var a = edges[e * 2]
            var b = edges[e * 2 + 1]
            if a < 0:
                continue
            var duplicate = False
            for f in range(e + 1, ne):
                if (edges[f * 2] == b and edges[f * 2 + 1] == a) or \
                               (edges[f * 2] == a and edges[f * 2 + 1] == b):
                    edges[f * 2] = -1
                    duplicate = True
                    break
            if duplicate:
                continue
            if nt >= cap:
                return -1
            if orient2(p, a, b, q) < 0.0:
                var tmp = a
                a = b
                b = tmp
            triangles[nt * 3] = a
            triangles[nt * 3 + 1] = b
            triangles[nt * 3 + 2] = q
            nt += 1
    var count = 0
    for t in range(nt):
        var a = triangles[t * 3]
        var b = triangles[t * 3 + 1]
        var c = triangles[t * 3 + 2]
        if a < n and b < n and c < n and orient2(p, a, b, c) > 1e-14:
            if count >= cap:
                return -1
            result[count * 3] = a
            result[count * 3 + 1] = b
            result[count * 3 + 2] = c
            count += 1
    return count


def in_sphere(p: FPtr, a: Int, b: Int, c: Int, d: Int, q: Int) -> Bool:
    var ax = p[a * 3]
    var ay = p[a * 3 + 1]
    var az = p[a * 3 + 2]
    var bx = p[b * 3]
    var by = p[b * 3 + 1]
    var bz = p[b * 3 + 2]
    var cx0 = p[c * 3]
    var cy0 = p[c * 3 + 1]
    var cz0 = p[c * 3 + 2]
    var dx0 = p[d * 3]
    var dy0 = p[d * 3 + 1]
    var dz0 = p[d * 3 + 2]
    var m00 = 2.0 * (bx - ax)
    var m01 = 2.0 * (by - ay)
    var m02 = 2.0 * (bz - az)
    var m10 = 2.0 * (cx0 - ax)
    var m11 = 2.0 * (cy0 - ay)
    var m12 = 2.0 * (cz0 - az)
    var m20 = 2.0 * (dx0 - ax)
    var m21 = 2.0 * (dy0 - ay)
    var m22 = 2.0 * (dz0 - az)
    var a_norm = ax * ax + ay * ay + az * az
    var r0 = bx * bx + by * by + bz * bz - a_norm
    var r1 = cx0 * cx0 + cy0 * cy0 + cz0 * cz0 - a_norm
    var r2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 - a_norm
    var det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)
    if det > -1e-18 and det < 1e-18:
        return False
    var cx = (r0 * (m11 * m22 - m12 * m21) - m01 * (r1 * m22 - m12 * r2) + m02 * (r1 * m21 - m11 * r2)) / det
    var cy = (m00 * (r1 * m22 - m12 * r2) - r0 * (m10 * m22 - m12 * m20) + m02 * (m10 * r2 - r1 * m20)) / det
    var cz = (m00 * (m11 * r2 - r1 * m21) - m01 * (m10 * r2 - r1 * m20) + r0 * (m10 * m21 - m11 * m20)) / det
    var dx = p[q * 3] - cx
    var dy = p[q * 3 + 1] - cy
    var dz = p[q * 3 + 2] - cz
    var ra = ax - cx
    var rb = ay - cy
    var rc = az - cz
    return dx * dx + dy * dy + dz * dz < (ra * ra + rb * rb + rc * rc) * 1.0000000001


def in_sphere_simd(p: FPtr, tetrahedra: IPtr, start: Int, q: Int) -> SIMD[DType.bool, W]:
    var base = iota[DType.int, W](start) * 4
    var a = tetrahedra.gather(base)
    var b = tetrahedra.gather(base + 1)
    var c = tetrahedra.gather(base + 2)
    var d = tetrahedra.gather(base + 3)
    var ax = p.gather(a * 3)
    var ay = p.gather(a * 3 + 1)
    var az = p.gather(a * 3 + 2)
    var bx = p.gather(b * 3)
    var by = p.gather(b * 3 + 1)
    var bz = p.gather(b * 3 + 2)
    var cx0 = p.gather(c * 3)
    var cy0 = p.gather(c * 3 + 1)
    var cz0 = p.gather(c * 3 + 2)
    var dx0 = p.gather(d * 3)
    var dy0 = p.gather(d * 3 + 1)
    var dz0 = p.gather(d * 3 + 2)
    var m00 = 2.0 * (bx - ax)
    var m01 = 2.0 * (by - ay)
    var m02 = 2.0 * (bz - az)
    var m10 = 2.0 * (cx0 - ax)
    var m11 = 2.0 * (cy0 - ay)
    var m12 = 2.0 * (cz0 - az)
    var m20 = 2.0 * (dx0 - ax)
    var m21 = 2.0 * (dy0 - ay)
    var m22 = 2.0 * (dz0 - az)
    var a_norm = ax * ax + ay * ay + az * az
    var r0 = bx * bx + by * by + bz * bz - a_norm
    var r1 = cx0 * cx0 + cy0 * cy0 + cz0 * cz0 - a_norm
    var r2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 - a_norm
    var det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)
    var valid = ~(det.gt(-1e-18) & det.lt(1e-18))
    var center_x = (r0 * (m11 * m22 - m12 * m21) - m01 * (r1 * m22 - m12 * r2) + m02 * (r1 * m21 - m11 * r2)) / det
    var center_y = (m00 * (r1 * m22 - m12 * r2) - r0 * (m10 * m22 - m12 * m20) + m02 * (m10 * r2 - r1 * m20)) / det
    var center_z = (m00 * (m11 * r2 - r1 * m21) - m01 * (m10 * r2 - r1 * m20) + r0 * (m10 * m21 - m11 * m20)) / det
    var qdx = p[q * 3] - center_x
    var qdy = p[q * 3 + 1] - center_y
    var qdz = p[q * 3 + 2] - center_z
    var ra = ax - center_x
    var rb = ay - center_y
    var rc = az - center_z
    return valid & (qdx * qdx + qdy * qdy + qdz * qdz).lt(
        (ra * ra + rb * rb + rc * rc) * 1.0000000001
    )


def sort_face(faces: IPtr, e: Int):
    var base = e * 3
    if faces[base] > faces[base + 1]:
        var tmp = faces[base]
        faces[base] = faces[base + 1]
        faces[base + 1] = tmp
    if faces[base + 1] > faces[base + 2]:
        var tmp = faces[base + 1]
        faces[base + 1] = faces[base + 2]
        faces[base + 2] = tmp
    if faces[base] > faces[base + 1]:
        var tmp = faces[base]
        faces[base] = faces[base + 1]
        faces[base + 1] = tmp


@export("mmp_delaunay3d")
def mmp_delaunay3d(
    points_addr: Int, n: Int, result_addr: Int, cap: Int, work_points_addr: Int,
    work_tetrahedra_addr: Int, faces_addr: Int
) abi("C") -> Int:
    if n < 4 or cap < 8:
        return 0
    if points_addr == 0 or result_addr == 0 or work_points_addr == 0 or \
            work_tetrahedra_addr == 0 or faces_addr == 0:
        return -2
    var points = FPtr(unsafe_from_address=points_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    var p = FPtr(unsafe_from_address=work_points_addr)
    var tetrahedra = IPtr(unsafe_from_address=work_tetrahedra_addr)
    var faces = IPtr(unsafe_from_address=faces_addr)
    var xmin = points[0]
    var xmax = xmin
    var ymin = points[1]
    var ymax = ymin
    var zmin = points[2]
    var zmax = zmin
    var point_values = n * 3
    var offset = 0
    while offset + W <= point_values:
        p.store(offset, points.load[width=W](offset))
        offset += W
    while offset < point_values:
        p[offset] = points[offset]
        offset += 1
    for i in range(n):
        if points[i * 3] < xmin: xmin = points[i * 3]
        if points[i * 3] > xmax: xmax = points[i * 3]
        if points[i * 3 + 1] < ymin: ymin = points[i * 3 + 1]
        if points[i * 3 + 1] > ymax: ymax = points[i * 3 + 1]
        if points[i * 3 + 2] < zmin: zmin = points[i * 3 + 2]
        if points[i * 3 + 2] > zmax: zmax = points[i * 3 + 2]
    var span = xmax - xmin
    if ymax - ymin > span: span = ymax - ymin
    if zmax - zmin > span: span = zmax - zmin
    if span <= 0.0: return 0
    var mx = (xmin + xmax) * 0.5
    var my = (ymin + ymax) * 0.5
    var mz = (zmin + zmax) * 0.5
    var s = 128.0 * span
    p[n * 3] = mx - s; p[n * 3 + 1] = my - s; p[n * 3 + 2] = mz - s
    p[(n + 1) * 3] = mx + s; p[(n + 1) * 3 + 1] = my - s; p[(n + 1) * 3 + 2] = mz + s
    p[(n + 2) * 3] = mx - s; p[(n + 2) * 3 + 1] = my + s; p[(n + 2) * 3 + 2] = mz + s
    p[(n + 3) * 3] = mx + s; p[(n + 3) * 3 + 1] = my + s; p[(n + 3) * 3 + 2] = mz - s
    tetrahedra[0] = n; tetrahedra[1] = n + 1; tetrahedra[2] = n + 2; tetrahedra[3] = n + 3
    var nt = 1
    for q in range(n):
        var nf = 0
        var write = 0
        var t = 0
        while t + W <= nt:
            var inside = in_sphere_simd(p, tetrahedra, t, q)
            for lane in range(W):
                var a = tetrahedra[(t + lane) * 4]
                var b = tetrahedra[(t + lane) * 4 + 1]
                var c = tetrahedra[(t + lane) * 4 + 2]
                var d = tetrahedra[(t + lane) * 4 + 3]
                if inside[lane]:
                    faces[nf * 3] = a; faces[nf * 3 + 1] = b; faces[nf * 3 + 2] = c; sort_face(faces, nf); nf += 1
                    faces[nf * 3] = a; faces[nf * 3 + 1] = b; faces[nf * 3 + 2] = d; sort_face(faces, nf); nf += 1
                    faces[nf * 3] = a; faces[nf * 3 + 1] = c; faces[nf * 3 + 2] = d; sort_face(faces, nf); nf += 1
                    faces[nf * 3] = b; faces[nf * 3 + 1] = c; faces[nf * 3 + 2] = d; sort_face(faces, nf); nf += 1
                else:
                    if write != t + lane:
                        tetrahedra[write * 4] = a
                        tetrahedra[write * 4 + 1] = b
                        tetrahedra[write * 4 + 2] = c
                        tetrahedra[write * 4 + 3] = d
                    write += 1
            t += W
        while t < nt:
            var a = tetrahedra[t * 4]
            var b = tetrahedra[t * 4 + 1]
            var c = tetrahedra[t * 4 + 2]
            var d = tetrahedra[t * 4 + 3]
            if in_sphere(p, a, b, c, d, q):
                faces[nf * 3] = a; faces[nf * 3 + 1] = b; faces[nf * 3 + 2] = c; sort_face(faces, nf); nf += 1
                faces[nf * 3] = a; faces[nf * 3 + 1] = b; faces[nf * 3 + 2] = d; sort_face(faces, nf); nf += 1
                faces[nf * 3] = a; faces[nf * 3 + 1] = c; faces[nf * 3 + 2] = d; sort_face(faces, nf); nf += 1
                faces[nf * 3] = b; faces[nf * 3 + 1] = c; faces[nf * 3 + 2] = d; sort_face(faces, nf); nf += 1
            else:
                if write != t:
                    tetrahedra[write * 4] = a
                    tetrahedra[write * 4 + 1] = b
                    tetrahedra[write * 4 + 2] = c
                    tetrahedra[write * 4 + 3] = d
                write += 1
            t += 1
        nt = write
        for f in range(nf):
            if faces[f * 3] < 0:
                continue
            var duplicate = False
            for g in range(f + 1, nf):
                if faces[f * 3] == faces[g * 3] and faces[f * 3 + 1] == faces[g * 3 + 1] and faces[f * 3 + 2] == faces[g * 3 + 2]:
                    faces[g * 3] = -1
                    duplicate = True
                    break
            if duplicate: continue
            if nt >= cap: return -1
            tetrahedra[nt * 4] = faces[f * 3]
            tetrahedra[nt * 4 + 1] = faces[f * 3 + 1]
            tetrahedra[nt * 4 + 2] = faces[f * 3 + 2]
            tetrahedra[nt * 4 + 3] = q
            nt += 1
    var count = 0
    for t in range(nt):
        var a = tetrahedra[t * 4]
        var b = tetrahedra[t * 4 + 1]
        var c = tetrahedra[t * 4 + 2]
        var d = tetrahedra[t * 4 + 3]
        if a < n and b < n and c < n and d < n:
            if count >= cap: return -1
            result[count * 4] = a; result[count * 4 + 1] = b
            result[count * 4 + 2] = c; result[count * 4 + 3] = d
            count += 1
    return count
