# Trefftz-EPW-2D

[![CI](https://github.com/BenJLatham/Trefftz_EPW/actions/workflows/ci.yml/badge.svg)](https://github.com/BenJLatham/Trefftz_EPW/actions)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15368677.svg)](https://doi.org/10.5281/zenodo.15368677)

We acknowledge support from NSF Grant DMS-2009366.
## Overview

**Trefftz-EPW-2D** is a Python package implementing a 2D plane-wave discontinuous Galerkin (Trefftz) solver for scattering by a circular inclusion with sign-changing permittivity. It supports:

- Graded or uniform meshes (disk or square domains) via Gmsh  
- Propagative and evanescent plane-wave basis functions (including Parolin distribution)  
- Assembly of the DG system with user-configurable flux parameters  
- Tikhonov regularization and choice of Pardiso or SciPy linear solvers  
- Post-processing: L²-error computation, field evaluation, and a command-line interface  

## Features

- **Modular design**: core solver class plus dedicated modules for numerics, meshing, assembly, and solving.   
- **Automated testing & CI**: pytest suite, Black formatting, flake8 linting.

## Installation

### Download a fixed snapshot (with DOI)

```bash
wget https://zenodo.org/record/15368677/files/Trefftz-EPW-2D.zip
unzip Trefftz-EPW-2D.zip
cd Trefftz-EPW-2D
````

### Or clone the GitHub repository

```bash
git clone https://github.com/BenJLatham/Trefftz_EPW.git
cd Trefftz_EPW
```

### Install in editable mode

```bash
pip install -e .
```

> **Requires** Python ≥ 3.8 and the dependencies:
> `numpy`, `scipy`, `meshio`, `gmsh`, `pypardiso`, `joblib`

## Quickstart

```python
import numpy as np
from Trefftz_EPW_2D import Trefftz2d

solver = Trefftz2d(
    k=5.0,
    epsilon=-2.0,
    order=96,
    zeta_distribution="Parolin",
    alpha=lambda x, y: 10**6,
    beta=lambda x, y: 10**6,
    delta=lambda x, y: 10**6,
    bc="robin"
)

solver.meshing(mesh="disk", h=0.05, R=1.0)
solver.create_basis_functions()
solver.assemble()
u = solver.solve(use_pardiso=False, alpha=None)

print("‖u‖₂ =", np.linalg.norm(u))
```

## CLI Usage

```bash
trefftzdg \
  --k 5.0 \
  --epsilon -2.0 \
  --order 9 \
  --zeta-distribution Parolin \
  --mesh disk \
  --h 0.05 \
  --R 1.0 \
  --use-pardiso \
  --alpha 1e-6 \
  --output solution.npz
```

Run `trefftzdg --help` for full options.

## Testing

```bash
pytest -q
```

## Citation

Please cite this software in your research:

```bibtex
@software{trefftzh-epw-2d_2025,
  author       = {Benjamin J. Latham},
  title        = {Trefftz-EPW-2D: Plane-wave DG solver for 2D disk scattering},
  doi          = {10.5281/zenodo.15368677},
  url          = {https://doi.org/10.5281/zenodo.15368677},
  year         = {2025},
}
```

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

```
```
