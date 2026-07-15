"""Registry of disorder-specific synthetic generation modules. Each entry
maps a target label to the module that generates it -- anorexia.py and
bulimia.py each cover two related labels that share almost all clinical
logic and differ mainly in weight or frequency targets (see their
docstrings), selected via the `label` argument to generate_group().
"""

from src.disorders import anorexia, arfid, binge_eating, bulimia, healthy, osfed_other
from src.features import LABELS

DISORDER_MODULES = {
    "none": healthy,
    "anorexia_nervosa": anorexia,
    "atypical_anorexia": anorexia,
    "bulimia_nervosa": bulimia,
    "atypical_bulimia_nervosa": bulimia,
    "binge_eating_disorder": binge_eating,
    "arfid": arfid,
    "osfed_other": osfed_other,
}
assert set(DISORDER_MODULES) == set(LABELS), "DISORDER_MODULES keys must match features.LABELS exactly"

# Balanced training distribution -- deliberately oversampled vs. true
# population prevalence (see PREVALENCE_LABEL_PROBS below) so every class
# has enough examples to learn from.
LABEL_PROBS = {
    "none": 0.45,
    "anorexia_nervosa": 0.09,
    "atypical_anorexia": 0.11,
    "bulimia_nervosa": 0.09,
    "atypical_bulimia_nervosa": 0.06,
    "binge_eating_disorder": 0.10,
    "arfid": 0.06,
    "osfed_other": 0.04,
}
assert set(LABEL_PROBS) == set(LABELS)

# Approximate true population prevalence, for a realistic-imbalance
# evaluation set only (never used for training). Order-of-magnitude
# estimates from: Hay et al. 2017 (atypical AN prevalence up to 3%);
# Lindvall Dahlgren et al. 2017 (AN ~1%, BN ~1-1.5%); Hail & Le Grange 2018
# (BN subthreshold rates); general ED epidemiology literature for BED/ARFID.
# Not a precise epidemiological model -- see README.md.
PREVALENCE_LABEL_PROBS = {
    "none": 0.90,
    "anorexia_nervosa": 0.010,
    "atypical_anorexia": 0.025,
    "bulimia_nervosa": 0.012,
    "atypical_bulimia_nervosa": 0.008,
    "binge_eating_disorder": 0.022,
    "arfid": 0.013,
    "osfed_other": 0.010,
}
assert set(PREVALENCE_LABEL_PROBS) == set(LABELS)
