# SPDX-License-Identifier: MPL-2.0
"""Shared runtime paths for standalone Instinct processes."""

import os
from pathlib import Path


def default_data_dir() -> str:
    """Resolve a user-owned data directory, never an installed package directory."""
    override = os.environ.get('INSTINCT_DATA_DIR') or os.environ.get('CE_DATA_DIR')
    if override:
        return str(Path(override).expanduser().absolute())
    base = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share')
    return str((base / 'instinct').expanduser().absolute())
