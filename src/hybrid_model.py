"""
hybrid_model.py
-----------------
Ties the whole pipeline together:

    raw student inputs
        -> Fuzzy Inference System (interpretable fuzzy score)
        -> Neural Networks trained 3 ways: Backprop / GA / PSO
        -> Hybrid fusion (weighted blend of fuzzy score + best NN prediction)
        -> Risk classification + human-readable recommendation

`HybridSystem.train()` runs once (at server startup) on the real dataset
and caches everything needed by the API: trained models, metrics and
convergence histories for the frontend charts.

`HybridSystem.predict()` is the fast, per-request inference path used by
the web form.
"""

import time
import numpy as np

from . import data_utils as du
from .fuzzy_logic import FuzzyPerformanceSystem
from .neural_network import SimpleNN
from .genetic_algorithm import run_ga
from .pso import run_pso
from .intervention import build_intervention


def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


class HybridSystem:
    def __init__(self):
        self.fis = FuzzyPerformanceSystem()
        self.nn_backprop = None
        self.nn_ga = None
        self.nn_pso = None
        self.best_model_name = None
        self.metrics = {}
        self.history = {"backprop": [], "ga": [], "pso": []}
        self.trained = False
        self.fuzzy_weight = 0.35  # blend weight in the final hybrid score

    # ------------------------------------------------------------------
    def train(self, hidden_dim=8):
        t0 = time.time()
        df = du.get_dataset()
        X = df[du.FEATURE_NAMES].to_numpy(dtype=float)
        y = df["performance_score"].to_numpy(dtype=float)

        Xn = du.normalize_features(X)
        yn = du.normalize_target(y)

        X_train, X_test, y_train, y_test = du.train_test_split(Xn, yn, test_ratio=0.2)

        # 1. Backpropagation baseline
        self.nn_backprop = SimpleNN(input_dim=X.shape[1], hidden_dim=hidden_dim, seed=1)
        self.history["backprop"] = self.nn_backprop.train_backprop(
            X_train, y_train, epochs=1500, lr=1.0
        )

        # 2. Genetic Algorithm
        self.nn_ga, self.history["ga"], _ = run_ga(
            X_train, y_train, hidden_dim=hidden_dim, pop_size=40, generations=60
        )

        # 3. Particle Swarm Optimization
        self.nn_pso, self.history["pso"], _ = run_pso(
            X_train, y_train, hidden_dim=hidden_dim, n_particles=30, iterations=60
        )

        # ---- evaluate all three on the held-out test set ----
        results = {}
        for name, model in [("backprop", self.nn_backprop),
                             ("ga", self.nn_ga),
                             ("pso", self.nn_pso)]:
            pred_n = model.predict(X_test)
            pred = du.denormalize_target(pred_n)
            true = du.denormalize_target(y_test)
            results[name] = {
                "mse": round(float(np.mean((pred - true) ** 2)), 3),
                "mae": round(_mae(true, pred), 3),
                "r2": round(_r2_score(true, pred), 4),
            }

        # also score the fuzzy system alone on the test set, for comparison
        fuzzy_preds = []
        for row in X_test:
            raw = self._unnormalize_row(row)
            fuzzy_preds.append(
                self.fis.infer(raw["attendance_pct"], raw["internal_marks"],
                                raw["assignment_score"], raw["study_hours"])["fuzzy_score"]
            )
        fuzzy_preds = np.array(fuzzy_preds)
        true = du.denormalize_target(y_test)
        results["fuzzy"] = {
            "mse": round(float(np.mean((fuzzy_preds - true) ** 2)), 3),
            "mae": round(_mae(true, fuzzy_preds), 3),
            "r2": round(_r2_score(true, fuzzy_preds), 4),
        }

        # hybrid = blend of fuzzy + best NN, evaluated on test set too
        self.best_model_name = min(
            ["backprop", "ga", "pso"], key=lambda k: results[k]["mse"]
        )
        best_nn = {"backprop": self.nn_backprop, "ga": self.nn_ga,
                   "pso": self.nn_pso}[self.best_model_name]
        nn_preds = du.denormalize_target(best_nn.predict(X_test))
        hybrid_preds = self.fuzzy_weight * fuzzy_preds + (1 - self.fuzzy_weight) * nn_preds
        results["hybrid"] = {
            "mse": round(float(np.mean((hybrid_preds - true) ** 2)), 3),
            "mae": round(_mae(true, hybrid_preds), 3),
            "r2": round(_r2_score(true, hybrid_preds), 4),
        }

        self.metrics = results
        self.trained = True
        self.train_time_sec = round(time.time() - t0, 2)
        self.n_samples = len(df)
        return self.metrics

    @staticmethod
    def _unnormalize_row(row):
        names = du.FEATURE_NAMES
        out = {}
        for i, name in enumerate(names):
            lo, hi = du.FEATURE_RANGES[name]
            out[name] = row[i] * (hi - lo) + lo
        return out

    # ------------------------------------------------------------------
    def predict(self, attendance_pct, internal_marks, assignment_score,
                study_hours, previous_cgpa):
        fuzzy_result = self.fis.infer(attendance_pct, internal_marks,
                                       assignment_score, study_hours)
        fuzzy_score = fuzzy_result["fuzzy_score"]

        raw = np.array([[attendance_pct, internal_marks, assignment_score,
                          study_hours, previous_cgpa]], dtype=float)
        Xn = du.normalize_features(raw)

        nn_predictions = {}
        for name, model in [("backprop", self.nn_backprop),
                             ("ga", self.nn_ga), ("pso", self.nn_pso)]:
            if model is not None:
                nn_predictions[name] = round(
                    float(du.denormalize_target(model.predict(Xn))[0]), 2
                )

        best_nn_score = nn_predictions.get(self.best_model_name, fuzzy_score)
        hybrid_score = round(
            self.fuzzy_weight * fuzzy_score + (1 - self.fuzzy_weight) * best_nn_score, 2
        )

        risk, performance_label = self._classify(hybrid_score)
        base = {
            "attendance": attendance_pct, "internal_marks": internal_marks,
            "assignment_score": assignment_score, "study_hours": study_hours,
            "previous_cgpa": previous_cgpa
        }
        intervention = build_intervention(base, hybrid_score, risk)
        risk_breakdown = self._risk_breakdown(base, hybrid_score)

        return {
            "fuzzy_score": fuzzy_score,
            "fired_rules": fuzzy_result["fired_rules"],
            "nn_predictions": nn_predictions,
            "best_model": self.best_model_name,
            "hybrid_score": hybrid_score,
            "performance_label": performance_label,
            "risk": risk,
            "recommendation": [a["action"] for a in intervention["actions"]],
            "intervention": intervention,
            "risk_breakdown": risk_breakdown,
            "memberships": fuzzy_result["memberships"],
        }


    @staticmethod
    def _risk_breakdown(data, score):
        factors = {
            "Attendance": max(0.0, min(1.0, (75-data["attendance"])/75)),
            "Internal Assessment": max(0.0, min(1.0, (60-data["internal_marks"])/60)),
            "Assignments": max(0.0, min(1.0, (60-data["assignment_score"])/60)),
            "Study Habit": max(0.0, min(1.0, (4-data["study_hours"])/4)),
            "Previous Performance": max(0.0, min(1.0, (6-data["previous_cgpa"])/6)),
        }
        return [{"factor": k, "risk_weight": round(v, 3)} for k,v in sorted(factors.items(), key=lambda x:x[1], reverse=True)]

    def predict_score_only(self, attendance_pct, internal_marks, assignment_score, study_hours, previous_cgpa):
        if not self.trained:
            raise RuntimeError("Models are not trained")
        raw = np.array([[attendance_pct, internal_marks, assignment_score, study_hours, previous_cgpa]], dtype=float)
        Xn = du.normalize_features(raw)
        best = {"backprop": self.nn_backprop, "ga": self.nn_ga, "pso": self.nn_pso}[self.best_model_name]
        nn_score = float(du.denormalize_target(best.predict(Xn))[0])
        fuzzy_score = self.fis.infer(attendance_pct, internal_marks, assignment_score, study_hours)["fuzzy_score"]
        return round(self.fuzzy_weight*fuzzy_score + (1-self.fuzzy_weight)*nn_score, 2)

    @staticmethod
    def _classify(score):
        if score >= 75:
            return "Low", "Excellent"
        elif score >= 60:
            return "Low", "Good"
        elif score >= 45:
            return "Medium", "Average"
        elif score >= 30:
            return "High", "Below Average"
        else:
            return "Very High", "Poor"

    @staticmethod
    def _recommend(attendance, internal, assignment, study_hours, cgpa, score):
        tips = []
        if attendance < 75:
            tips.append("Improve class attendance — aim for at least 75%.")
        if internal < 60:
            tips.append("Focus on internal assessments; consider extra practice tests.")
        if assignment < 60:
            tips.append("Submit assignments more consistently and start them earlier.")
        if study_hours < 4:
            tips.append("Increase daily study hours (target 4-5 hrs/day).")
        if cgpa < 6:
            tips.append("Revisit fundamentals from previous semesters with a tutor or study group.")

        if not tips:
            if score >= 75:
                tips.append("Excellent trajectory — maintain current habits and consider mentoring peers.")
            else:
                tips.append("Keep up consistent effort across attendance, assessments and study time.")

        return tips


# a module-level singleton the Flask app can import and reuse
system = HybridSystem()


if __name__ == "__main__":
    import json
    s = HybridSystem()
    m = s.train()
    print(json.dumps(m, indent=2))
    print("best NN optimizer:", s.best_model_name)
    demo = s.predict(72, 68, 80, 3, 7.4)
    print(json.dumps(demo, indent=2))

