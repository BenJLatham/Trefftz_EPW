import numpy as np
from Trefftz_EPW_2D.assembly import assemble_system


def test_assemble_system_empty():
    # No lines → zero matrix & zero RHS
    lines = np.empty((0, 8))
    P = np.array([1], dtype=int)
    cumsum_P = np.array([0], dtype=int)
    A, rhs = assemble_system(
        lines=lines,
        P=P,
        cumsum_P=cumsum_P,
        diagonal_block=lambda e, line, flip: np.array([[1 + 0j]]),
        off_diagonal_block=lambda *args: np.array([[0j]]),
        assemble_rhs_vec=lambda *args: np.array([0j]),
        total_dofs=1,
    )
    assert A.shape == (1, 1)
    assert rhs.shape == (1,)
    assert A.nnz == 0
    assert rhs[0] == 0
