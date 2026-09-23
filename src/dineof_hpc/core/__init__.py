# Copyright (C) 2025 Arthur Coque, RBINS
# Copyright (C) 2026 Julien Masson, LOG UMR 8187
# Distributed under the terms of the GNU General Public License version 3 (GPLv3).

"""Core numerical algorithms, spectral solvers, and NetCDF I/O pipelines."""

from .main import (
    dineof,
    compute_dineof_oi_error_step,
    select_cv_blocks_multi,
)
from .io_utils import (
    load_netcdf_masked,
    run_reconstruction_pipeline,
    fit_quantile_transfer_training,
    stream_reconstruction_and_oi_netcdf,
    export_diagnostics_and_plots,
)

__all__ = [
    "dineof",
    "compute_dineof_oi_error_step",
    "select_cv_blocks_multi",
    "load_netcdf_masked",
    "run_reconstruction_pipeline",
    "fit_quantile_transfer_training",
    "stream_reconstruction_and_oi_netcdf",
    "export_diagnostics_and_plots",
]