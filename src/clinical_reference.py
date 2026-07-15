"""Shared clinical reference calculations used by both data generation
(src/disorders/) and feature engineering (src/features.py), so the two stay
in sync by construction instead of via duplicated formulas."""

import numpy as np


def reference_median_bmi(age, sex):
    """Coarse population-median BMI-for-age/sex, used to compute percent
    median BMI (%mBMI) -- the metric Sick Enough favors over raw BMI for
    adolescents/atypical presentations (Ch. 1, p.851-856). Simple linear
    approximation of CDC growth-chart medians, not the real charts."""
    age_c = np.clip(age, 10, 60)
    pediatric = 15.5 + (age_c - 10) * ((21.0 - 15.5) / 10.0)
    adult = np.where(np.asarray(sex) == "M", 24.5, 23.5)
    return np.where(age_c >= 20, adult, pediatric)
