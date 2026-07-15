"""Avoidant/Restrictive Food Intake Disorder (ARFID).

Sick Enough Ch. 14: restrictive intake driven by sensory sensitivity, lack
of interest in food, or fear of aversive consequences (e.g. choking,
vomiting) -- WITHOUT the body image disturbance or fear of weight gain that
define AN/AAN. That absence is the actual DSM-5 exclusion criterion
separating ARFID from the anorexia spectrum, and is deliberately
constructed here as the main discriminating signal: restrictive_intake_score
stays high while fear_of_weight_gain_score/body_image_disturbance_score
stay low -- an inversion of the AN/AAN pattern rather than a lower-severity
version of it.
"""

import numpy as np

from src.disorders import common


def generate_group(rng, age, sex, height_cm, label: str) -> dict:
    n = len(age)
    row = common.baseline_row_defaults(rng, n)

    intake_severity = rng.beta(5, 2, size=n) * 10
    row["restrictive_intake_score"] = common.behavioral_score(rng, intake_severity, n)
    # Defining exclusion criterion: no weight/shape driver behind the restriction.
    low_severity = rng.beta(1.5, 6, size=n) * 10
    row["fear_of_weight_gain_score"] = common.behavioral_score(rng, low_severity, n, noise_sd=0.8)
    row["body_image_disturbance_score"] = common.behavioral_score(rng, low_severity, n, noise_sd=0.8)
    row["exercise_compulsion_score"] = common.behavioral_score(rng, low_severity, n, noise_sd=0.8)

    row["food_variety_count"] = np.clip(rng.normal(6, 3, size=n), 1, 15)  # markedly narrowed diet
    row["fear_of_aversive_consequences"] = rng.random(n) < np.clip(0.5 * (intake_severity / 10), 0, 0.6)
    row["sensory_sensitivity_score"] = common.behavioral_score(rng, intake_severity, n)

    # ARFID can produce real undernutrition/growth faltering without any
    # weight-driven behavior -- weight suppression here is a consequence of
    # intake severity, not something pursued intentionally.
    wt = common.weight_trajectory(
        rng, age, sex, height_cm, n,
        pct_median_bmi_mean=88, pct_median_bmi_sd=14, pct_median_bmi_clip=(55, 130),
        suppression_mean=10, suppression_sd=8, suppression_clip=(0, 35),
        duration_shape=3, duration_scale=3, duration_clip=(1, 48),
    )
    row["current_weight_kg"] = wt["current_weight_kg"]
    row["highest_past_weight_kg"] = wt["highest_past_weight_kg"]
    row["weight_loss_duration_months"] = wt["weight_loss_duration_months"]

    resilience = rng.uniform(0, 1, size=n)
    effective_stress = common.physiologic_stress(intake_severity, wt["_weight_suppression_pct"], resilience)
    mal_factor = common.malnutrition_factor(wt["_pct_median_bmi"], resilience)

    row.update(common.vitals_and_labs(rng, n, effective_stress, mal_factor, np.zeros(n), np.zeros(n, dtype=bool)))
    row.update(common.exam_findings(rng, n, effective_stress))
    row.update(common.purging_exam_findings(rng, n, np.zeros(n)))
    row["amenorrhea"] = common.amenorrhea(rng, n, sex, effective_stress, wt["_weight_suppression_pct"])

    return row
