"""No eating disorder -- population-normal baseline. Implemented as a
disorder module for architectural consistency (every label goes through the
same generate_group interface via the DISORDER_MODULES registry), even
though there's no disorder-specific clinical logic here.
"""

import numpy as np

from src.disorders import common


def generate_group(rng, age, sex, height_cm, label: str) -> dict:
    n = len(age)
    row = common.baseline_row_defaults(rng, n)

    latent_severity = rng.beta(1.5, 6, size=n) * 10
    row["fear_of_weight_gain_score"] = common.behavioral_score(rng, latent_severity, n)
    row["body_image_disturbance_score"] = common.behavioral_score(rng, latent_severity, n)
    row["restrictive_intake_score"] = common.behavioral_score(rng, latent_severity, n)
    row["exercise_compulsion_score"] = common.behavioral_score(rng, latent_severity, n)

    wt = common.weight_trajectory(
        rng, age, sex, height_cm, n,
        pct_median_bmi_mean=102, pct_median_bmi_sd=14, pct_median_bmi_clip=(65, 170),
        suppression_mean=2, suppression_sd=3, suppression_clip=(0, 15),
        duration_shape=2, duration_scale=1.5, duration_clip=(0.5, 12),
    )
    row["current_weight_kg"] = wt["current_weight_kg"]
    row["highest_past_weight_kg"] = wt["highest_past_weight_kg"]
    row["weight_loss_duration_months"] = wt["weight_loss_duration_months"]

    resilience = rng.uniform(0, 1, size=n)
    effective_stress = common.physiologic_stress(latent_severity, wt["_weight_suppression_pct"], resilience)
    mal_factor = common.malnutrition_factor(wt["_pct_median_bmi"], resilience)

    row.update(common.vitals_and_labs(rng, n, effective_stress, mal_factor, np.zeros(n), np.zeros(n, dtype=bool)))
    row.update(common.exam_findings(rng, n, effective_stress))
    row.update(common.purging_exam_findings(rng, n, np.zeros(n)))
    row["amenorrhea"] = common.amenorrhea(rng, n, sex, effective_stress, wt["_weight_suppression_pct"])

    return row
