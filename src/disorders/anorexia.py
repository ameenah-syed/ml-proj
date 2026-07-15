"""Anorexia nervosa (AN) and atypical anorexia nervosa (AAN).

Sick Enough Ch. 1 (p.9-10): AN and AAN patients "engage in all the same
behaviors and have equally severe body image distortions and fears" as each
other -- weight status is the DSM-5 differentiator, not illness severity.
Behavioral scores are therefore drawn from the same distribution for both
labels here.

AN has two subtypes, restricting and binge/purge (p.818-829; the
purge-subtype description at p.4561-4563 is explicit that patients
"restrict calories and purge them, with or without bingeing," and still
meet the underweight criterion). Subtype is modeled as purging_behaviors /
binge_behaviors flags rather than a separate label.
"""

import numpy as np

from src.disorders import common


def generate_group(rng, age, sex, height_cm, label: str) -> dict:
    n = len(age)
    row = common.baseline_row_defaults(rng, n)

    latent_severity = rng.beta(6, 1.8, size=n) * 10
    row["fear_of_weight_gain_score"] = common.behavioral_score(rng, latent_severity, n)
    row["body_image_disturbance_score"] = common.behavioral_score(rng, latent_severity, n)
    row["restrictive_intake_score"] = common.behavioral_score(rng, latent_severity, n)
    row["exercise_compulsion_score"] = common.behavioral_score(rng, latent_severity, n)

    if label == "anorexia_nervosa":
        wt = common.weight_trajectory(
            rng, age, sex, height_cm, n,
            pct_median_bmi_mean=76, pct_median_bmi_sd=7, pct_median_bmi_clip=(50, 110),
            suppression_mean=18, suppression_sd=8, suppression_clip=(5, 45),
            duration_shape=4, duration_scale=2.6, duration_clip=(1, 42),
        )
    else:  # atypical_anorexia
        wt = common.weight_trajectory(
            rng, age, sex, height_cm, n,
            pct_median_bmi_mean=98, pct_median_bmi_sd=10, pct_median_bmi_clip=(60, 150),
            suppression_mean=20, suppression_sd=9, suppression_clip=(5, 45),
            duration_shape=4, duration_scale=3.3, duration_clip=(1, 48),
        )
    row["current_weight_kg"] = wt["current_weight_kg"]
    row["highest_past_weight_kg"] = wt["highest_past_weight_kg"]
    row["weight_loss_duration_months"] = wt["weight_loss_duration_months"]

    # ~35% purge subtype (with or without bingeing); rest restricting subtype.
    purge = common.purging_profile(rng, n, purge_prob=0.35)
    row["purging_behaviors"] = purge["purging_behaviors"]
    row["compensatory_behavior_type"] = purge["compensatory_behavior_type"]
    binge = common.binge_profile(rng, n, freq_shape=1.5, freq_scale=0.6, freq_clip=(0, 6))
    # Bingeing in AN only co-occurs with the purge subtype.
    row["binge_frequency_per_week"] = np.where(purge["purging_behaviors"], binge["binge_frequency_per_week"], 0.0)
    row["binge_behaviors"] = row["binge_frequency_per_week"] >= 0.1

    resilience = rng.uniform(0, 1, size=n)
    effective_stress = common.physiologic_stress(latent_severity, wt["_weight_suppression_pct"], resilience)
    mal_factor = common.malnutrition_factor(wt["_pct_median_bmi"], resilience)

    row.update(common.vitals_and_labs(rng, n, effective_stress, mal_factor, purge["_purging_intensity"], purge["_volume_depleted"]))
    row.update(common.exam_findings(rng, n, effective_stress))
    row.update(common.purging_exam_findings(rng, n, purge["_purging_intensity"]))
    row["amenorrhea"] = common.amenorrhea(rng, n, sex, effective_stress, wt["_weight_suppression_pct"])

    return row
