"""Utility helpers for the EPJDS Definition Study code release.

This file exists primarily to satisfy lightweight helper imports used by the
analysis scripts (e.g., set_working_directory()).

Note: For the publication repository, keep this module *free of secrets*.
"""

from __future__ import annotations

import os
from pathlib import Path


def set_working_directory() -> str:
    """Set the working directory to the repository root.

    The scripts in this project were developed as executable entrypoints.
    For reproducibility, we normalise execution by changing the current
    working directory to the directory containing this file.

    Returns
    -------
    str
        Absolute path to the repository root.
    """
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    return str(repo_root)
