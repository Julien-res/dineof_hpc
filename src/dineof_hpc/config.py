# Copyright (C) 2026 Julien Masson, LOG UMR 8187
# Distributed under the terms of the GNU General Public License version 3 (GPLv3).

"""Read DINEOF parameter profiles from a TOML file (used by the example scripts)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:  # Python >= 3.11
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9 / 3.10
    import tomli as tomllib

DEFAULT_CONFIG = Path(__file__).with_name("dineof_params.toml")


def load_dine_opts(
    path: str | os.PathLike | None = None,
    profile: str = "conservative",
    **overrides: Any,
) -> dict[str, Any]:
    """Return the ``dineof_kwargs`` dictionary of one profile of a TOML file.

    Parameters
    ----------
    path : str or path-like, optional
        TOML file. Defaults to the ``DINEOF_CONFIG`` environment variable, then to
        ``dineof_params.toml`` next to this module.
    profile : str
        Name of the table to read (``"conservative"`` or ``"fast"`` in the shipped file).
    **overrides
        Values that replace those of the file, e.g. ``nev=30``.

    Notes
    -----
    An absent ``nev`` (or ``nev = 0``) means automatic estimation (``None``).
    """
    cfg_path = Path(path or os.environ.get("DINEOF_CONFIG") or DEFAULT_CONFIG)
    with open(cfg_path, "rb") as fh:
        profiles = tomllib.load(fh)
    if profile not in profiles:
        raise ValueError(f"Unknown profile {profile!r} in {cfg_path}; available: {sorted(profiles)}")
    opts = dict(profiles[profile])
    opts.update(overrides)
    if not opts.get("nev"):
        opts["nev"] = None
    return opts
