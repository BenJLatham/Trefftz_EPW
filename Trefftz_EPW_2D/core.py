"""
core.py

Trefftz2d class orchestrator: high-level interface composing mesh generation, basis creation,
assembly, solution, and post-processing.
"""

import meshio
import numpy as np

from .meshing import generate_disk_mesh, generate_square_mesh
from .mesh_utils import make_lines
from .pw_utils import build_pw_params
from .assembly import assemble_system
from .solver import solve_system
from .analytic import sum_u
from .quadrature import gaussian_quadrature_points_and_weights
from .assembly import compute_raw_diag, compute_raw_offdiag

__all__ = ["Trefftz2d"]


class Trefftz2d:
    """
    Trefftz Plane-Wave Discontinuous Galerkin solver for 2D scattering.

    This class orchestrates mesh creation, plane-wave basis generation, system assembly,
    linear solve (with optional regularization), and field evaluation.
    """

    def __init__(
        self,
        k,
        epsilon,
        order,
        zeta_distribution="None",
        alpha=lambda x, y: 0.5,
        beta=lambda x, y: 0.5,
        delta=lambda x, y: 0.5,
        bc="robin",
    ):
        """
        Parameters:
          k                 : float   – wavenumber in exterior medium
          epsilon           : float   – relative permittivity of inclusion
          order             : int     – target number of plane waves per element
          zeta_distribution : str     – mode for evanescent distribution
          alpha, beta, delta: callables (x,y) -> float – flux penalty functions
          bc                : str     – boundary condition type
        """
        self.k = k
        self.epsilon = epsilon
        self.order = order
        self.zeta_distribution = zeta_distribution
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.bc = bc

        # placeholders for later
        self.mesh = None
        self.lines = None
        self.centroids = None
        self.epsilon_values = None
        self.epsilon_inv_values = None
        self.D_mats = None
        self.P = None
        self.cumsum_P = None
        self.result_matrix = None
        self.rhs = None
        self.solution = None

    def meshing(self, mesh_type, **kwargs):
        """
        Generate and load a mesh.

        mesh_type : str – 'disk' or 'square'
        kwargs    : parameters for mesh generator (h, R, L, filename)
        """
        if mesh_type == "disk":
            filename = kwargs.get("filename", None)
            generate_disk_mesh(h=kwargs["h"], R=kwargs["R"], filename=filename)
        elif mesh_type == "square":
            generate_square_mesh(
                h=kwargs["h"],
                R=kwargs["R"],
                L=kwargs["L"],
                filename=kwargs.get("filename", "square.msh"),
            )
        else:
            raise ValueError(f"Unknown mesh_type {mesh_type}")

        # read back the mesh
        self.mesh = meshio.read(filename or f"{mesh_type}.msh")
        # extract connectivity and geometry
        self.lines = make_lines(self.mesh)
        self.centroids = np.mean(
            self.mesh.points[self.mesh.cells_dict["triangle"]], axis=1
        )
        # set permittivity arrays per element
        N = len(self.centroids)
        self.epsilon_values = np.full(N, self.epsilon)
        self.epsilon_inv_values = 1.0 / self.epsilon_values

    def create_basis_functions(self):
        """
        Build plane-wave directions and lambdas for each element,
        storing per-element metadata in self.D_mats.
        """
        self.D_mats = {}
        for idx, center in enumerate(self.centroids):
            thetas, zetas = build_pw_params(
                cx=center[0],
                cy=center[1],
                P_elem=self.order,
                eps_val=self.epsilon,
                zmode=self.zeta_distribution,
                k=self.k,
            )
            # store directions and normalization coefs
            dir_vecs = np.column_stack([np.cos(thetas), np.sin(thetas)])
            norm_coeff = 1.0  # or compute as needed
            self.D_mats[idx] = {"dir_vecs": dir_vecs, "normalization_coeff": norm_coeff}
        # build P and cumsum_P arrays
        P_list = [d["dir_vecs"].shape[0] for d in self.D_mats.values()]
        self.P = np.array(P_list, dtype=int)
        self.cumsum_P = np.concatenate([[0], np.cumsum(self.P)[:-1]])

    def assemble(self):
        """
        Assemble the global system matrix and RHS vector.
        """
        total_dofs = int(np.sum(self.P))
        self.result_matrix, self.rhs = assemble_system(
            lines=self.lines,
            P=self.P,
            cumsum_P=self.cumsum_P,
            diagonal_block=lambda e, line, flip: compute_raw_diag(
                e,
                line,
                flip,
                self.centroids,
                self.epsilon_values,
                self.epsilon_inv_values,
                self.D_mats,
                self.k,
                self.alpha,
                self.beta,
                self.delta,
                self.bc,
            ),
            off_diagonal_block=lambda e1, e2, line, flip: compute_raw_offdiag(
                e1,
                e2,
                line,
                flip,
                self.centroids,
                self.epsilon_values,
                self.epsilon_inv_values,
                self.D_mats,
                self.k,
                self.alpha,
                self.beta,
                self.delta,
                self.bc,
            ),
            assemble_rhs_vec=lambda e, line, flip: None,  # implement if needed
            total_dofs=total_dofs,
        )

    def solve(self, use_pardiso=False, alpha_reg=None):
        """
        Solve the assembled system with optional regularization.
        """
        self.solution = solve_system(
            A=self.result_matrix, b=self.rhs, use_pardiso=use_pardiso, alpha=alpha_reg
        )

    def evaluate_field(self, r, theta):
        """
        Evaluate the analytic total field at polar points (r, theta).
        """
        return sum_u(
            N=max(self.P),
            r=r,
            k=self.k,
            theta=theta,
            epsm=self.epsilon,
            theta0=np.pi / 2,
        )

    def quadrature_rule(self, n):
        """
        Return quadrature points and weights for reference triangle.
        """
        return gaussian_quadrature_points_and_weights(n)
