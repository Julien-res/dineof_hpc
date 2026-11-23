# -*- coding: utf-8 -*-
"""Example configuration for the DINEOF-HPC reconstruction pipeline.

This example is intentionally dataset-agnostic. It expects a NetCDF file with
one time-varying variable and a compatible NetCDF mask containing a variable
named ``mask``. Missing observations must be encoded as ``NaN``.

Run this file after editing the paths below and the parameters in
``dineof_params.toml``::

    python example.py

The ``conservative`` profile is recommended for final reconstructions. The
``fast`` profile is intended for previews and parameter exploration.
"""

from __future__ import annotations

import os
from pathlib import Path

from dineof_hpc.config import load_dine_opts
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
#    The parameters (nev, alpha, toliter...) live in dineof_params.toml, next to
#    this script; edit that file, or point DINEOF_CONFIG to your own copy.
#    Keep "conservative" for final results and use "fast" for previews.
# -----------------------------------------------------------------------------
PROFILE = "conservative"
DINE_OPTS = {name: load_dine_opts(profile=name) for name in ("conservative", "fast")}


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