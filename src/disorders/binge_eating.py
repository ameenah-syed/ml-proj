"""Binge eating disorder (BED).

Sick Enough Ch. 10 ("Binge Eating Disorder (BED) and Weight Stigma"):
recurrent binge episodes with loss of control, WITHOUT the compensatory
behaviors that define BN, often in higher-weight patients whose medical
complications the chapter argues are frequently misattributed to weight
itself rather than worked up as an eating disorder. DSM-5's frequency
threshold mirrors BN (>=1x/week for 3 months) but there is no purging, so
purging-driven physiology (electrolytes, orthostasis) stays at
population-normal levels here.
"""

import numpy as np

from src.disorders import common


def generate_group(rng, age, sex, height_cm, label: str) -> dict:
    n = len(age)
    row = common.baseline_row_defaults(rng, n)

    latent_severity = rng.beta(5, 2.5, size=n) * 10
    row["fear_of_weight_gain_score"] = common.behavioral_score(rng, latent_severity * 0.6, n)
    row["body_image_disturbance_score"] = common.behavioral_score(rng, latent_severity, n)
    row["restrictive_intake_score"] = common.behavioral_score(rng, latent_severity * 0.3, n)
    row["exercise_compulsion_score"] = common.behavioral_score(rng, latent_severity * 0.4, n)

    wt = common.weight_trajectory(
        rng, age, sex, height_cm, n,
        pct_median_bmi_mean=125, pct_median_bmi_sd=20, pct_median_bmi_clip=(90, 200),
        suppression_mean=3, suppression_sd=4, suppression_clip=(0, 20),
        duration_shape=2, duration_scale=1.5, duration_clip=(0.5, 12),
    )
    row["current_weight_kg"] = wt["current_weight_kg"]
    row["highest_past_weight_kg"] = wt["highest_past_weight_kg"]
    row["weight_loss_duration_months"] = wt["weight_loss_duration_months"]

    row["purging_behaviors"] = np.zeros(n, dtype=bool)
    row["compensatory_behavior_type"] = np.full(n, "none")
    binge = common.binge_profile(rng, n, freq_shape=4, freq_scale=0.7, freq_clip=(1, 10))
    row["binge_frequency_per_week"] = binge["binge_frequency_per_week"]
    row["binge_behaviors"] = np.ones(n, dtype=bool)
    row["loss_of_control_eating"] = np.ones(n, dtype=bool)

    resilience = rng.uniform(0, 1, size=n)
    # Not starvation-driven -- mild, weight-stigma/metabolic complications
    # rather than the restriction physiology used elsewhere.
    effective_stress = np.clip(latent_severity / 30, 0, 0.3) * (1 - 0.5 * resilience)
    mal_factor = np.zeros(n)

    row.update(common.vitals_and_labs(rng, n, effective_stress, mal_factor, np.zeros(n), np.zeros(n, dtype=bool)))
    row.update(common.exam_findings(rng, n, effective_stress * 0.2))
    row.update(common.purging_exam_findings(rng, n, np.zeros(n)))
    row["amenorrhea"] = common.amenorrhea(rng, n, sex, effective_stress * 0.3, wt["_weight_suppression_pct"])

    return row
