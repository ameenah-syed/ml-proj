"""OSFED, other specified feeding and eating disorder -- subthreshold
catch-all (Ch. 1's general OSFED framing, p.830-831, under which atypical
AN and atypical BN also formally fall, but which this project splits out
into their own named labels since the book gives them dedicated
discussion). This module covers everything else: moderate severity across
behavioral axes, without meeting the full frequency/duration/weight
thresholds for any single named disorder above.
"""

from src.disorders import common


def generate_group(rng, age, sex, height_cm, label: str) -> dict:
    n = len(age)
    row = common.baseline_row_defaults(rng, n)

    latent_severity = rng.beta(3, 4, size=n) * 10  # moderate, below full-threshold AN/BN/BED severity
    row["fear_of_weight_gain_score"] = common.behavioral_score(rng, latent_severity, n)
    row["body_image_disturbance_score"] = common.behavioral_score(rng, latent_severity, n)
    row["restrictive_intake_score"] = common.behavioral_score(rng, latent_severity, n)
    row["exercise_compulsion_score"] = common.behavioral_score(rng, latent_severity, n)

    wt = common.weight_trajectory(
        rng, age, sex, height_cm, n,
        pct_median_bmi_mean=95, pct_median_bmi_sd=15, pct_median_bmi_clip=(70, 150),
        suppression_mean=9, suppression_sd=6, suppression_clip=(0, 30),
        duration_shape=3, duration_scale=2, duration_clip=(1, 30),
    )
    row["current_weight_kg"] = wt["current_weight_kg"]
    row["highest_past_weight_kg"] = wt["highest_past_weight_kg"]
    row["weight_loss_duration_months"] = wt["weight_loss_duration_months"]

    purge = common.purging_profile(rng, n, purge_prob=0.15)
    row["purging_behaviors"] = purge["purging_behaviors"]
    row["compensatory_behavior_type"] = purge["compensatory_behavior_type"]
    binge = common.binge_profile(rng, n, freq_shape=1.5, freq_scale=0.3, freq_clip=(0, 3))
    row["binge_frequency_per_week"] = binge["binge_frequency_per_week"]
    row["binge_behaviors"] = row["binge_frequency_per_week"] >= 0.1

    resilience = rng.uniform(0, 1, size=n)
    effective_stress = common.physiologic_stress(latent_severity, wt["_weight_suppression_pct"], resilience)
    mal_factor = common.malnutrition_factor(wt["_pct_median_bmi"], resilience)

    row.update(common.vitals_and_labs(rng, n, effective_stress, mal_factor, purge["_purging_intensity"], purge["_volume_depleted"]))
    row.update(common.exam_findings(rng, n, effective_stress))
    row.update(common.purging_exam_findings(rng, n, purge["_purging_intensity"]))
    row["amenorrhea"] = common.amenorrhea(rng, n, sex, effective_stress, wt["_weight_suppression_pct"])

    return row
