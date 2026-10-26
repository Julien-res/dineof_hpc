# -*- coding: utf-8 -*-
"""Example configuration for the DINEOF-HPC reconstruction pipeline.

This example is intentionally dataset-agnostic. It expects a NetCDF file with
one time-varying variable and a compatible NetCDF mask containing a variable
named ``mask``. Missing observations must be encoded as ``NaN``.

Run this file after editing the paths and options below::

    python example.py

The ``conservative`` profile is recommended for final reconstructions. The
``fast`` profile is intended for previews and parameter exploration.
"""

from __future__ import annotations

import os
from pathlib import Path

from dineof_hpc.core.io_utils import run_reconstruction_pipeline


# -----------------------------------------------------------------------------
# 1. User paths and dataset options.
#    Replace these example paths and values with settings for your dataset.
# -----------------------------------------------------------------------------
MASK_FILE = Path("data/water_mask.nc")

JOBS = [
    {
        "name": "example",
        "input": Path("data/input.nc"),
        "output": Path("output/dineof_example"),
        "var_name": "data",
        "threshold": None,           # Values <= threshold are treated as missing.
        "vmax": None,                # Optional upper bound for reconstructed values.
        "apply_log": True,           # Use for strictly positive environmental data.
        "use_climatology": True,     # Remove and restore the temporal pixel mean.
        "apply_qmap": False,         # Opt-in: no measurable benefit in benchmarks (see README §5).
    }
]


# -----------------------------------------------------------------------------
# 2. DINEOF profiles.
#    Select the profile below. Keep "conservative" for final results and use
#    "fast" for previews or parameter exploration.
# -----------------------------------------------------------------------------
PROFILE = "conservative"

DINE_OPTS = {
    "conservative": {
        "nev": None,                 # Estimate the rank from the variance target.
        "toliter": 0.003,            # Relative convergence tolerance.
        "nitemax": 300,              # Maximum iterations allowed per EOF mode.
        "alpha": 0.005,              # Temporal diffusion strength.
        "cv_patch_size": 11,         # Side length of validation patches in pixels.
        "cv_max_points": 500_000,    # Maximum number of validation points.
        "n_cv_runs": 3,              # Number of independent validation patch sets.
        "cv_patience": 3,            # Modes allowed after the CV minimum.
        "keep_observations": True,   # Preserve genuine observations exactly.
        "subsample_ratio": 0.1,     # Use 10% of pixels for covariance estimation.
        "seed": 42,                  # Make sampling and results reproducible.
        "verbose": True,             # Print progress and convergence information.
        "debug": True,               # Include detailed timing information.
    },
    "fast": {
        "nev": 15,                   # Avoid automatic rank estimation.
        "toliter": 0.01,             # Stop earlier with a looser tolerance.
        "nitemax": 50,               # Use fewer iterations per EOF mode.
        "alpha": 0.005,              # Keep the temporal filter unchanged.
        "cv_patch_size": 11,         # Keep the same spatial validation scale.
        "cv_max_points": 50_000,     # Evaluate fewer validation points.
        "n_cv_runs": 1,              # Use one validation patch set.
        "cv_patience": 1,            # Stop soon after the CV minimum.
        "keep_observations": True,   # Preserve genuine observations exactly.
        "subsample_ratio": 0.1,     # Keep covariance computation compact.
        "seed": 42,                  # Keep comparisons reproducible.
        "verbose": True,             # Keep progress visible.
        "debug": False,              # Skip timing instrumentation overhead.
    },
}


# Optional thread settings. Remove this block if a scheduler configures them.
NB_THREADS = os.environ.get("OMP_NUM_THREADS", "4")
SCRATCH_PATH = Path("output/tmp")
os.environ.setdefault("OMP_NUM_THREADS", NB_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", NB_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", NB_THREADS)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(SCRATCH_PATH))
SCRATCH_PATH.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run every reconstruction job declared in ``JOBS``."""
    if PROFILE not in DINE_OPTS:
        raise ValueError(f"Unknown DINEOF profile: {PROFILE!r}")

    for job in JOBS:
        print(f"\n{'=' * 50}")
        print(f"Starting DINEOF reconstruction: {job['name']}")
        print(f"Profile: {PROFILE}")
        print(f"{'=' * 50}")

        run_reconstruction_pipeline(
            input_nc=str(job["input"]),
            mask_file=str(MASK_FILE),
            var_name=job["var_name"],
            out_dir=str(job["output"]),
            threshold=job["threshold"],
            vmax=job["vmax"],
            min_valid_ratio=0.01,
            apply_log=job["apply_log"],
            use_climatology=job["use_climatology"],
            apply_qmap=job["apply_qmap"],
            qmap_nb_quantiles=100,
            use_disk=False,            # Set True when RAM is limited.
            solver="auto",             # Select the spectral solver automatically.
            dineof_kwargs=DINE_OPTS[PROFILE],
        )


if __name__ == "__main__":
    main()