"""
assembly.py

Assembly routines for the Trefftz PWDG method: pure helper functions for local block computations
and a single‐pass global assembler.
"""

import numpy as np
from scipy.sparse import coo_matrix
from Trefftz_EPW_2D.mesh_utils import normal_jacobian
from Trefftz_EPW_2D.helpers import complex_sqrt, psi


__all__ = ["compute_raw_diag", "compute_raw_offdiag", "assemble_system"]


def compute_raw_diag(
    elem,
    line,
    flip,
    centroids,
    epsilon_values,
    epsilon_inv_values,
    D_mats,
    k,
    alpha,
    beta,
    delta,
    bc_type,
):
    """
    Compute the local diagonal block for element `elem` on `line`,
    exactly as in Trefftz2d.diagonal_block but as a pure function.
    """
    # 1) pull out all the data we need
    dir_vecs = D_mats[elem]["dir_vecs"]  # shape (P,2)
    eps = epsilon_values[elem]
    eps_inv = epsilon_inv_values[elem]
    # geometry
    jac, n0, n1 = normal_jacobian(line)
    a = np.array(line[:2])
    b = np.array(line[2:4])
    normals = np.array([n0, n1])
    if flip:
        normals = -normals
    # relative vectors along the edge and from element center
    ba = b - a
    center = centroids[elem]
    # compute dot products for the psi‐kernel
    diffs = dir_vecs[None, :, :] - np.conj(dir_vecs)[:, None, :]  # (P,P,2)
    dot_ba = np.sum(diffs * ba[None, None, :], axis=2)  # (P,P)
    dot_rc = np.sum(diffs * (a - center)[None, None, :], axis=2)  # (P,P)

    # build the “plane‐wave integral” kernel
    sqrt_eps = complex_sqrt(eps)
    norm_coeff = D_mats[elem]["normalization_coeff"]
    PRI = (
        norm_coeff
        * np.exp(1j * k * sqrt_eps * dot_rc)
        * jac
        * psi(1j * k * sqrt_eps * dot_ba)
    )  # (P,P)

    # normal‐derivatives of the basis
    norm_deriv = 1j * k * sqrt_eps * (dir_vecs @ normals)  # (P,)

    # 2) boundary face?
    if int(line[5]) == -1:
        δ = delta(0, 0)
        t1 = (1 - δ) * 1j * k * PRI
        t2 = δ * PRI * norm_deriv[None, :]
        t3 = (δ - 1) * PRI * np.conj(norm_deriv)[:, None]
        t4 = (
            -δ
            * (eps_inv / (1j * k))
            * (np.conj(norm_deriv)[:, None] * norm_deriv[None, :])
            * PRI
        )
        return t1 + t2 + t3 + t4
    # 3) interior face
    else:
        α = alpha(0, 0)
        β = beta(0, 0)
        A = -α * 1j * k * PRI
        B = (
            eps_inv**2
            * (β / (1j * k))
            * (np.conj(norm_deriv)[:, None] * norm_deriv[None, :])
            * PRI
        )
        T1 = -0.5 * eps_inv * PRI * np.conj(norm_deriv)[:, None]
        T2 = 0.5 * eps_inv * PRI * norm_deriv[None, :]
        return A + B + T1 + T2


def compute_raw_offdiag(
    elem1,
    elem2,
    line,
    flip,
    centroids,
    epsilon_values,
    epsilon_inv_values,
    D_mats,
    k,
    alpha,
    beta,
    delta,
    bc_type,
):
    """
    Compute the local off‐diagonal block coupling elem1→elem2 on `line`.
    """
    # 1) Unpack geometry & directional data
    D1 = D_mats[elem1]["dir_vecs"]  # (P1,2)
    D2 = D_mats[elem2]["dir_vecs"]  # (P2,2)
    c1 = centroids[elem1]  # (2,)
    c2 = centroids[elem2]
    eps_inv1 = epsilon_inv_values[elem1]
    eps_inv2 = epsilon_inv_values[elem2]
    sqrt1 = complex_sqrt(epsilon_values[elem1])
    sqrt2 = complex_sqrt(epsilon_values[elem2])

    # 2) Geometry: normal & jacobian
    jac, n0, n1 = normal_jacobian(line)
    a, b = np.array(line[:2]), np.array(line[2:4])
    normals = np.array([n0, n1])
    if flip:
        normals = -normals

    # 3) Build the phase matrix Φ of shape (P2, P1)
    φ1 = 1j * k * sqrt1 * (D1 @ (a - c1))  # (P1,)
    φ2 = -1j * k * sqrt2 * (np.conj(D2) @ (a - c2))  # (P2,)
    Φ = φ2[:, None] + φ1[None, :]  # (P2, P1)

    # 4) Directional differences & edge‐dot
    diffs = sqrt1 * D1[None, :, :] - sqrt2 * np.conj(D2)[:, None, :]  # (P2,P1,2)
    ba = b - a
    dot_ba = np.sum(diffs * ba[None, None, :], axis=2)  # (P2, P1)

    # 5) Principal integral PRI (shape P2×P1)
    coeff = D_mats[elem2]["normalization_coeff"]
    PRI = coeff * jac * np.exp(Φ) * psi(1j * k * dot_ba)

    # 6) Normal derivatives on each side
    nd1 = 1j * k * sqrt1 * (D1 @ normals)  # (P1,)
    nd2 = np.conj(1j * k * sqrt2 * (D2 @ (-normals)))  # (P2,)

    # 7) Broadcast for matrix arithmetic
    nd1e = nd1[None, :]  # (1,P1)
    nd2e = nd2[:, None]  # (P2,1)

    # 8) Flux parameters
    α = alpha(0, 0)
    β = beta(0, 0)

    # 9) Build the off-diagonal block
    termA = α * 1j * k * PRI
    termB = (eps_inv1 * eps_inv2 * β / (1j * k)) * (nd2e * nd1e) * PRI
    term1 = -0.5 * eps_inv2 * PRI * nd2e
    term2 = -0.5 * eps_inv1 * PRI * nd1e

    return termA + termB + term1 + term2


def assemble_system(
    lines, P, cumsum_P, diagonal_block, off_diagonal_block, assemble_rhs_vec, total_dofs
):
    """
    Assemble global sparse system matrix A and RHS vector in a single pass,
    using user‐provided local block functions.

    Parameters:
      lines                : ndarray, shape (M,8)
        Mesh line definitions: [x1,y1,x2,y2,elem1,elem2,tag,inside_flag]
      P                    : ndarray, shape (N,)
        Number of basis functions per element.
      cumsum_P             : ndarray, shape (N,)
        Cumulative DOF start indices for each element.
      diagonal_block       : callable(elem, line, flip) -> ndarray(P,P)
      off_diagonal_block   : callable(elem_i, elem_j, line, flip) -> ndarray(Pi,Pj)
      assemble_rhs_vec     : callable(elem, line, flip) -> ndarray(P,)
      total_dofs           : int
        Total number of global DOFs (sum(P)).

    Returns:
      A   : csr_matrix, shape (total_dofs, total_dofs)
      rhs : ndarray, shape (total_dofs,)
    """
    rows, cols, data = [], [], []
    rhs = np.zeros(total_dofs, dtype=complex)

    for line in lines:
        e1 = int(line[4])
        e2 = int(line[5])
        flip1 = (line[7] == 0 and e1 == line[5]) or (line[7] == 1 and e1 == line[4])
        flip2 = not flip1

        # --- Diagonal block for element e1 ---
        P1 = P[e1]
        i1 = cumsum_P[e1]
        D11 = diagonal_block(e1, line, flip1)
        idx1 = np.arange(P1)
        r11 = np.repeat(idx1, P1)
        c11 = np.tile(idx1, P1)
        rows.extend((i1 + r11).tolist())
        cols.extend((i1 + c11).tolist())
        data.extend(D11.ravel().tolist())

        if e2 == -1:
            # Boundary face → RHS only
            v = assemble_rhs_vec(e1, line, flip1)
            rhs[i1 : i1 + P1] += v
        else:
            # Diagonal block for element e2
            P2 = P[e2]
            i2 = cumsum_P[e2]
            D22 = diagonal_block(e2, line, flip2)
            idx2 = np.arange(P2)
            r22 = np.repeat(idx2, P2)
            c22 = np.tile(idx2, P2)
            rows.extend((i2 + r22).tolist())
            cols.extend((i2 + c22).tolist())
            data.extend(D22.ravel().tolist())

            # Off‐diagonal couplings
            O12 = off_diagonal_block(e1, e2, line, flip1)
            O21 = off_diagonal_block(e2, e1, line, flip2)
            r12 = np.repeat(idx1, P2)
            c12 = np.tile(idx2, P1)
            rows.extend((i1 + r12).tolist())
            cols.extend((i2 + c12).tolist())
            data.extend(O12.ravel().tolist())
            r21 = np.repeat(idx2, P1)
            c21 = np.tile(idx1, P2)
            rows.extend((i2 + r21).tolist())
            cols.extend((i1 + c21).tolist())
            data.extend(O21.ravel().tolist())

    A = coo_matrix((data, (rows, cols)), shape=(total_dofs, total_dofs)).tocsr()
    return A, rhs
