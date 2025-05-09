
# Trefftz-EPW-2D

[![CI](https://github.com/YourUserName/Trefftz_EPW/actions/workflows/ci.yml/badge.svg)](https://github.com/YourUserName/Trefftz_EPW/actions)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15368677.svg)](https://doi.org/10.5281/zenodo.15368677)

## Overview

**Trefftz-EPW-2D** is a Python package implementing a 2D plane-wave discontinuous Galerkin (Trefftz) solver for scattering by a circular inclusion with sign-changing permittivity. It supports:

- Graded or uniform meshes (disk or square domains) via Gmsh  
- Propagative and evanescent plane-wave basis functions (including Parolin distribution)  
- Assembly of the DG system with user-configurable flux parameters  
- Tikhonov regularization and choice of Pardiso or SciPy linear solvers  
- Post-processing: L²-error computation, field evaluation, and a command-line interface  

## Features

- **Modular design**: core solver class plus dedicated modules for numerics, meshing, assembly, and solving.  
- **No PyPI needed**: simply clone or download a snapshot (see Zenodo DOI) and run.  
- **Automated testing & CI**: pytest suite, Black formatting, flake8 linting.

## Installation

You can download a fixed archived snapshot with DOI 10.5281/zenodo.15368677 from Zenodo:

wget https://zenodo.org/record/15368677/files/Trefftz-EPW-2D.zip
unzip Trefftz-EPW-2D.zip
cd Trefftz-EPW-2D


Or clone the GitHub repository:

```bash
git clone https://github.com/YourUserName/Trefftz_EPW.git
cd Trefftz_EPW

Then install in editable mode so you can edit and rerun:

pip install -e .

(Requires Python ≥3.8, plus dependencies: numpy, scipy, meshio, gmsh, pypardiso, joblib.)

Quickstart

import numpy as np
from Trefftz_EPW_2D import Trefftz2d

solver = Trefftz2d(
    k=5.0,
    epsilon=-2.0,
    order=9,
    zeta_distribution="Parolin",
    alpha=lambda x,y:0.5,
    beta=lambda x,y:0.5,
    delta=lambda x,y:0.5,
    bc="robin"
)

solver.meshing(mesh="disk", h=0.05, R=1.0)
solver.create_basis_functions()
solver.assemble()
u = solver.solve(use_pardiso=False, alpha=None)

print("‖u‖₂ =", np.linalg.norm(u))


Or run directly from the command line:

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


Testing
pytest -q

If you use Trefftz-EPW-2D in your research, please cite:

@software{trefftzh-epw-2d_2025,
  author       = {Benjamin J. Latham},
  title        = {Trefftz-EPW-2D: Plane-wave DG solver for 2D disk scattering},
  doi          = {10.5281/zenodo.15368677},
  url          = {https://doi.org/10.5281/zenodo.15368677},
  year         = {2025},
}


