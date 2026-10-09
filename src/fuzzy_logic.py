"""
fuzzy_logic.py
--------------
A self-contained Mamdani Fuzzy Inference System (FIS) built from scratch
(no external fuzzy library dependency) covering:

    - Fuzzy sets & triangular membership functions
    - Fuzzification of crisp inputs
    - A fuzzy rule base (IF...AND...THEN...)
    - Mamdani-style inference (min for AND, max for aggregation)
    - Centroid defuzzification

Inputs (crisp, real-world units):
    attendance_pct    0-100
    internal_marks    0-100
    assignment_score  0-100
    study_hours       0-12  (hours/day)

Output:
    performance (fuzzy score) 0-100
"""

import numpy as np


def trimf(x, abc):
    """Triangular membership function, abc = (a, b, c).

    When a == b (or b == c) the function degenerates into a left/right
    "shoulder" set (e.g. Low=(0,0,50) or High=(65,100,100)): the degenerate
    edge imposes no constraint of its own (it's treated as already fully
    satisfied) and the remaining slope alone shapes the membership curve.
    """
    a, b, c = abc
    left = 1.0 if a == b else (x - a) / (b - a)
    right = 1.0 if b == c else (c - x) / (c - b)
    return float(max(min(left, right, 1.0), 0.0))


# ---------------------------------------------------------------------
# Membership function definitions (Low / Medium / High) per input,
# and (Poor / Average / Good / Excellent) for the output.
# ---------------------------------------------------------------------

MF_ATTENDANCE = {
    "Low": (0, 0, 50),
    "Medium": (30, 55, 80),
    "High": (65, 100, 100),
}
MF_MARKS = {  # used for both internal_marks and assignment_score (0-100)
    "Low": (0, 0, 45),
    "Medium": (30, 55, 75),
    "High": (60, 100, 100),
}
MF_STUDY_HOURS = {
    "Low": (0, 0, 3),
    "Medium": (2, 5, 8),
    "High": (6, 12, 12),
}
MF_OUTPUT = {
    "Poor": (0, 0, 35),
    "Average": (20, 45, 65),
    "Good": (50, 70, 85),
    "Excellent": (75, 100, 100),
}

OUTPUT_DOMAIN = np.linspace(0, 100, 201)  # discretised universe for defuzzification


def fuzzify(value, mf_dict):
    return {label: trimf(value, abc) for label, abc in mf_dict.items()}


# ---------------------------------------------------------------------
# Rule base. Each rule: (attendance_label, internal_label, assignment_label,
# study_label_or_None, output_label). `None` means "don't care" for that
# antecedent so the rule fires on the remaining antecedents only.
# ---------------------------------------------------------------------

RULES = [
    ("Low", "Low", "Low", None, "Poor"),
    ("Low", "Low", "Medium", None, "Poor"),
    ("Low", "Medium", "Low", None, "Poor"),
    ("Low", "High", "High", None, "Average"),
    ("Medium", "Low", "Low", None, "Poor"),
    ("Medium", "Medium", "Medium", None, "Average"),
    ("Medium", "Medium", "High", None, "Good"),
    ("Medium", "High", "Medium", None, "Good"),
    ("Medium", "High", "High", None, "Good"),
    ("High", "Low", "Low", None, "Average"),
    ("High", "Medium", "Medium", None, "Good"),
    ("High", "Medium", "High", None, "Good"),
    ("High", "High", "Medium", None, "Excellent"),
    ("High", "High", "High", None, "Excellent"),
    (None, None, None, "High", "Good"),       # strong study habit boosts confidence
    (None, None, None, "Low", "Average"),
    ("Low", None, None, "Low", "Poor"),
    ("High", "High", "High", "High", "Excellent"),
]


class FuzzyPerformanceSystem:
    """Mamdani FIS mapping (attendance, internal_marks, assignment, study_hours)
    -> a fuzzy performance score in [0, 100]."""

    def infer(self, attendance_pct, internal_marks, assignment_score, study_hours):
        att = fuzzify(attendance_pct, MF_ATTENDANCE)
        intm = fuzzify(internal_marks, MF_MARKS)
        asg = fuzzify(assignment_score, MF_MARKS)
        study = fuzzify(study_hours, MF_STUDY_HOURS)

        aggregated = np.zeros_like(OUTPUT_DOMAIN)
        fired_rules = []

        for att_l, intm_l, asg_l, study_l, out_l in RULES:
            strengths = []
            if att_l is not None:
                strengths.append(att[att_l])
            if intm_l is not None:
                strengths.append(intm[intm_l])
            if asg_l is not None:
                strengths.append(asg[asg_l])
            if study_l is not None:
                strengths.append(study[study_l])
            if not strengths:
                continue
            firing = min(strengths)
            if firing <= 0:
                continue
            fired_rules.append({
                "rule": f"IF attendance={att_l} AND internal={intm_l} AND "
                        f"assignment={asg_l} AND study={study_l} THEN {out_l}",
                "firing_strength": round(firing, 3),
            })
            a, b, c = MF_OUTPUT[out_l]
            clipped = np.minimum(firing, [trimf(x, (a, b, c)) for x in OUTPUT_DOMAIN])
            aggregated = np.maximum(aggregated, clipped)

        # Centroid defuzzification
        if aggregated.sum() == 0:
            score = 50.0  # fallback: neutral if nothing fired
        else:
            score = float(np.sum(OUTPUT_DOMAIN * aggregated) / np.sum(aggregated))

        return {
            "fuzzy_score": round(score, 2),
            "memberships": {
                "attendance": att,
                "internal_marks": intm,
                "assignment_score": asg,
                "study_hours": study,
            },
            "fired_rules": fired_rules,
        }


if __name__ == "__main__":
    fis = FuzzyPerformanceSystem()
    result = fis.infer(72, 68, 80, 3)
    print(result["fuzzy_score"])
    for r in result["fired_rules"]:
        print(r)
