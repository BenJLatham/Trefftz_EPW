"""
solver.py

Solvers for Trefftz PWDG linear systems: Tikhonov regularization, Pardiso & SciPy backends.
"""

import time
import numpy as np
import pypardiso
from scipy.sparse import eye, bmat, issparse
from scipy.sparse.linalg import spsolve

__all__ = ["solve_system"]


def solve_system(A, b, use_pardiso=False, alpha=None):
    """
    Solve A x = b with optional Tikhonov regularization and choice of solver.

    Parameters:
      A           : csr_matrix (complex)
                    System matrix.
      b           : array-like or sparse (complex)
                    Right-hand side vector.
      use_pardiso : bool
                    If True, solve using pypardiso; else use SciPy's spsolve.
      alpha       : float or None
                    Tikhonov regularization parameter; if not None, solves
                    (A^H A + alpha I) x = A^H b.

    Returns:
      x : ndarray (complex)
          Solution vector.
    """
    # Ensure b is a 1D numpy array
    if issparse(b):
        b_vec = b.toarray().ravel()
    else:
        b_vec = np.array(b).ravel()

    # Apply Tikhonov regularization
    if alpha is not None:
        A_H = A.getH()
        reg_mat = A_H.dot(A) + alpha * eye(A.shape[1], format="csr")
        reg_rhs = A_H.dot(b_vec)
    else:
        reg_mat = A
        reg_rhs = b_vec

    # Choose solver
    if use_pardiso:
        # Build real equivalent system for complex A
        A_r = reg_mat.real
        A_i = reg_mat.imag
        M = bmat([[A_r, -A_i], [A_i, A_r]], format="csr")
        b_r = reg_rhs.real
        b_i = reg_rhs.imag
        B = np.concatenate([b_r, b_i])

        start = time.time()
        sol_real = pypardiso.spsolve(M, B)
        elapsed = time.time() - start
        print(f"Solving with Pardiso took {elapsed:.4f} seconds.")

        n = b_r.size
        x_r = sol_real[:n]
        x_i = sol_real[n:]
        x = x_r + 1j * x_i
    else:
        start = time.time()
        x = spsolve(reg_mat, reg_rhs)
        elapsed = time.time() - start
        print(f"Solving with scipy.spsolve took {elapsed:.4f} seconds.")

    return x
