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
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="trefftzdg",
        description="Run the 2D plane-wave DG Trefftz solver"
    )
    # core solver parameters
    parser.add_argument("--k", type=float, required=True,
                        help="Wavenumber in exterior medium")
    parser.add_argument("--epsilon", type=float, required=True,
                        help="Relative permittivity of inclusion")
    parser.add_argument("--order", type=int, required=True,
                        help="Target number of plane waves per element")
    parser.add_argument("--zeta-distribution", choices=["None","Interface","Parolin"],
                        default="None", help="Evanescent mode distribution")
    parser.add_argument("--bc", choices=["robin","dirichlet"], default="robin",
                        help="Boundary condition type")
    # mesh parameters
    parser.add_argument("--mesh", choices=["disk","square"], default="disk",
                        help="Mesh shape")
    parser.add_argument("--h", type=float, required=True,
                        help="Mesh size parameter")
    parser.add_argument("--R", type=float, default=1.0,
                        help="Radius for disk mesh")
    parser.add_argument("--L", type=float,
                        help="Side length for square mesh (required if mesh=square)")
    # solver options
    parser.add_argument("--use-pardiso", action="store_true",
                        help="Use Pardiso solver instead of SciPy")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Tikhonov regularization parameter")
    # output
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save solution vector (NumPy .npz)")

    args = parser.parse_args()

    # Build the solver
    solver = Trefftz2d(
        k=args.k,
        epsilon=args.epsilon,
        order=args.order,
        zeta_distribution=args.zeta_distribution,
        alpha=lambda x,y: args.alpha or 0.5,
        beta=lambda  x,y: args.alpha or 0.5,
        delta=lambda x,y: args.alpha or 0.5,
        bc=args.bc
    )

    # Mesh
    if args.mesh == "disk":
        solver.meshing(h=args.h, R=args.R)
    else:
        if args.L is None:
            parser.error("--L is required for square mesh")
        solver.meshing(h=args.h, R=args.R, L=args.L)

    # Build basis, assemble, and solve
    solver.create_basis_functions()
    solver.assemble()
    solution = solver.solve(use_pardiso=args.use_pardiso, alpha=args.alpha)

    # Output
    if args.output:
        import numpy as _np
        _np.savez(args.output, solution=solution)
        print(f"Solution saved to {args.output}")
    else:
        # Just print a summary
        print("Solution vector (first 10 entries):", solution[:10])
        print("Norm of solution:", np.linalg.norm(solution))
