"""
meshing.py

Mesh generation routines using Gmsh and mesh caching.
"""

import gmsh

__all__ = ["generate_disk_mesh", "generate_square_mesh"]


def generate_disk_mesh(h=0.1, R=1.5, filename=None):
    """
    Generate a 2D disk mesh with inner radius 1 and outer radius R using mesh size h.
    Writes to 'disk.msh' or the provided filename.

    Parameters:
      h        : float – target mesh size.
      R        : float – outer radius.
      filename : str or None – if given, rename 'disk.msh' to this filename.
    """
    gmsh.initialize()
    gmsh.model.add("Disk")
    # Define points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, h)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, h)
    p3 = gmsh.model.geo.addPoint(R, 0, 0, h)
    p4 = gmsh.model.geo.addPoint(-1, 0, 0, h)
    p5 = gmsh.model.geo.addPoint(-R, 0, 0, h)
    # Define arcs
    inner1 = gmsh.model.geo.addCircleArc(p2, p1, p4)
    inner2 = gmsh.model.geo.addCircleArc(p4, p1, p2)
    outer1 = gmsh.model.geo.addCircleArc(p3, p1, p5)
    outer2 = gmsh.model.geo.addCircleArc(p5, p1, p3)
    # Curve loops & surfaces
    inner_loop = gmsh.model.geo.addCurveLoop([inner1, inner2])
    outer_loop = gmsh.model.geo.addCurveLoop([outer1, outer2, -inner1, -inner2])
    gmsh.model.geo.addPlaneSurface([inner_loop])
    gmsh.model.geo.addPlaneSurface([outer_loop])
    # Physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, "metal")
    gmsh.model.addPhysicalGroup(2, [2], 2)
    gmsh.model.setPhysicalName(2, 2, "vacuum")
    # Synchronize & mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    out = filename or "disk.msh"
    gmsh.write(out)
    gmsh.finalize()


def generate_square_mesh(h, R, L, filename="square_mesh.msh"):  # noqa: C901
    """
    Generate a 2D mesh of a circle of radius R cut by a square of side L using mesh size h.
    Writes to the given filename.

    Parameters:
      h        : float – target mesh size.
      R        : float – circle radius.
      L        : float – square side length.
      filename : str – output .msh filename.
    """
    gmsh.initialize()
    gmsh.model.add("Square")
    # Define circle (outer)
    _ = gmsh.model.occ.addPoint(0, 0, 0)
    circ = gmsh.model.occ.addCircle(0, 0, 0, R)
    # Define square
    half = L / 2
    pts = [
        gmsh.model.occ.addPoint(-half, -half, 0),
        gmsh.model.occ.addPoint(half, -half, 0),
        gmsh.model.occ.addPoint(half, half, 0),
        gmsh.model.occ.addPoint(-half, half, 0),
    ]
    lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    sq_loop = gmsh.model.occ.addCurveLoop(lines)
    sq_surf = gmsh.model.occ.addPlaneSurface([sq_loop])
    circ_loop = gmsh.model.occ.addCurveLoop([circ])
    circle_surf = gmsh.model.occ.addPlaneSurface([circ_loop, sq_loop])
    gmsh.model.occ.synchronize()
    # Physical groups
    gmsh.model.addPhysicalGroup(2, [sq_surf], name="metal")
    gmsh.model.addPhysicalGroup(2, [circle_surf], name="vacuum")
    gmsh.model.addPhysicalGroup(1, [circ], name="outer")
    gmsh.model.addPhysicalGroup(1, lines, name="inner")
    # Mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
