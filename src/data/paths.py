#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

# Project root inferred from src/ layout
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Default AMLEC dataset pickle path with env override
_DEFAULT_AMLEC_PICKLE_PATH = DATA_DIR / "AMLEC_dataset.pkl"
_ENV_AMLEC_PICKLE_PATH = os.environ.get("AMLEC_PICKLE_PATH")
AMLEC_PICKLE_PATH = (
    Path(_ENV_AMLEC_PICKLE_PATH).expanduser() if _ENV_AMLEC_PICKLE_PATH else _DEFAULT_AMLEC_PICKLE_PATH
)

