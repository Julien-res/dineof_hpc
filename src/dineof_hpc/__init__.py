# src/dineof_hpc/__init__.py
"""DINEOF-HPC: High-performance spatio-temporal reconstruction for geophysical datasets."""

from dineof_hpc.__about__ import (
    __author__,
    __description__,
    __license__,
    __title__,
    __version__,
)
from dineof_hpc.core.io_utils import load_netcdf_masked, run_reconstruction_pipeline
from dineof_hpc.core.main import dineof

__all__ = [
    "dineof",
    "run_reconstruction_pipeline",
    "load_netcdf_masked",
    "__title__",
    "__version__",
    "__author__",
    "__license__",
    "__description__",
]