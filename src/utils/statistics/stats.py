#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def run_linear_model(long_df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    model = smf.ols("psi ~ C(modality) * C(first_mode)", data=long_df).fit()
    summary_text = model.summary().as_text()
    effects = {
        "modality_effect": (model.params.get("C(modality)[T.Adaptive]", np.nan)),
        "order_effect": (model.params.get("C(first_mode)[T.Adaptive]", np.nan)),
        "interaction_effect": (model.params.get("C(modality)[T.Adaptive]:C(first_mode)[T.Adaptive]", np.nan)),
    }
    return summary_text, effects

