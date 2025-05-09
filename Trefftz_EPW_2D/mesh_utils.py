"""
mesh_utils.py

Mesh utilities for PWDG: point‐in‐triangle tests, line/edge extraction, and related helpers.
"""

import numpy as np
from scipy.spatial import KDTree

__all__ = [
    "is_point_on_edge_or_vertex",
    "is_point_inside_triangle",
    "find_triangles_containing_point",
    "normal_jacobian",
    "normalize_line",
    "merge_duplicate_lines",
    "make_lines",
]


def is_point_on_edge_or_vertex(pt, v1, v2, tol=1e-8):
    """
    Check if point pt lies exactly on the segment [v1, v2], including endpoints.

    Parameters:
      pt  : array‐like, shape (2,) – point coordinates (x, y).
      v1  : array‐like, shape (2,) – first endpoint of segment.
      v2  : array‐like, shape (2,) – second endpoint of segment.
      tol : float – numerical tolerance.

    Returns:
      bool – True if pt lies on the segment within tolerance.
    """
    pt = np.asarray(pt)
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    # Check colinearity via cross‐product magnitude
    cross = np.abs(np.cross(v2 - v1, pt - v1))
    if cross > tol:
        return False
    # Check within bounding box
    dot = np.dot(pt - v1, v2 - v1)
    if dot < -tol or dot > np.dot(v2 - v1, v2 - v1) + tol:
        return False
    return True


def is_point_inside_triangle(pt, triangle_pts):
    """
    Determine if a point is strictly inside a triangle.

    Parameters:
      pt           : array‐like, shape (2,) – point (x, y).
      triangle_pts : array‐like, shape (3,2) – vertices of the triangle.

    Returns:
      bool – True if pt lies inside the triangle (excluding edges).
    """
    pt = np.asarray(pt)
    a, b, c = map(np.asarray, triangle_pts)
    # Compute barycentric coordinates
    v0 = c - a
    v1 = b - a
    v2 = pt - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u > 0) and (v > 0) and (u + v < 1)


def find_triangles_containing_point(mesh, point, k=10):
    """
    Find mesh triangles that contain a given point by KDTree candidate search.

    Parameters:
      mesh  : meshio.Mesh – must have .points (N×2 or more) and .cells_dict['triangle']
      point : array‐like, shape (2,) – query point (x, y).
      k     : int – number of nearest triangle centroids to check.

    Returns:
      list of int – indices of triangles from mesh.cells_dict['triangle'] containing point.
    """
    vertices = mesh.points[:, :2]
    cells = mesh.cells_dict["triangle"]
    centroids = np.mean(vertices[cells], axis=1)
    kdtree = KDTree(centroids)
    _, idxs = kdtree.query(point, k=min(k, len(centroids)))
    candidates = np.atleast_1d(idxs)
    found = []
    for idx in candidates:
        tri = vertices[cells[idx]]
        if is_point_inside_triangle(point, tri):
            found.append(idx)
    return found


def normal_jacobian(line):
    """
    Compute the length (Jacobian) and outward normal for a line segment.

    Parameters:
      line : sequence – [x1, y1, x2, y2, ...]

    Returns:
      (jacobian, nx, ny): float and components of unit normal.
    """
    x1, y1, x2, y2 = line[:4]
    jacobian = np.hypot(x2 - x1, y2 - y1)
    normal = np.array([y1 - y2, x2 - x1])
    normal = normal / np.linalg.norm(normal)
    return jacobian, normal[0], normal[1]


def normalize_line(line, return_swapped=False):
    """
    Sort line endpoints so (x1,y1) < (x2,y2) lexicographically.

    Parameters:
      line           : sequence – [x1, y1, x2, y2, ...]
      return_swapped : bool – if True, also return a flag if swapped occurred.

    Returns:
      normalized line (and optional swapped flag).
    """
    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    if (x1 > x2) or (x1 == x2 and y1 > y2):
        if return_swapped:
            return [x2, y2, x1, y1] + list(line[4:]), True
        return [x2, y2, x1, y1] + list(line[4:])
    else:
        if return_swapped:
            return [x1, y1, x2, y2] + list(line[4:]), False
        return [x1, y1, x2, y2] + list(line[4:])


def merge_duplicate_lines(lines, triangles, points):
    """
    Merge lines that share the same endpoints, populating neighbor info and flags.

    Parameters:
      lines     : array-like – each row [x1,y1,x2,y2, tri1, tri2, tag, inside_flag]
      triangles : array of triangle vertex indices
      points    : array of mesh point coordinates

    Returns:
      ndarray – merged line array with unique edges.
    """
    line_dict = {}
    step = 1e-5
    for line in lines:
        norm_line, swapped = normalize_line(line, return_swapped=True)
        key = tuple(norm_line[:4])
        if key in line_dict:
            entry = line_dict[key]
            entry[5] = line[4]
            if entry[6] != line[6]:
                entry[6] = 10
        else:
            line_dict[key] = list(norm_line)
        # check inside flag via stepping
        mid = np.array(
            [(norm_line[0] + norm_line[2]) / 2, (norm_line[1] + norm_line[3]) / 2]
        )
        jac, nx, ny = normal_jacobian(norm_line)
        step_point = mid + step * np.array([nx, ny])
        tri_pts = points[triangles[int(line_dict[key][4])]][:, :2]
        line_dict[key][7] = int(is_point_inside_triangle(step_point, tri_pts))
    return np.array(list(line_dict.values()))


def make_lines(mesh):
    """
    Extract all mesh edges as lines with adjacency info.

    Parameters:
      mesh : meshio.Mesh – with .points and .cells_dict['triangle']

    Returns:
      ndarray – rows [x1,y1,x2,y2, tri1, tri2, tag, inside_flag]
    """
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    num_triangles = len(triangles)
    all_lines = []
    triangle_tags = np.zeros(num_triangles, dtype=int)
    for idx, tri in enumerate(triangles):
        for edge in [(0, 1), (1, 2), (2, 0)]:
            v1, v2 = tri[edge[0]], tri[edge[1]]
            x1, y1 = points[v1]
            x2, y2 = points[v2]
            all_lines.append([x1, y1, x2, y2, idx, -1, triangle_tags[idx], 0])
    lines = np.array(all_lines)
    return merge_duplicate_lines(lines, triangles, points)
