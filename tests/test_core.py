"""Lightweight offline verification for the core SCOA modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hybrid_model import HybridSystem
from src.study_optimizer import optimize_ga, optimize_pso


def main():
    system = HybridSystem()
    metrics = system.train()
    assert set(metrics) >= {"backprop", "ga", "pso", "fuzzy", "hybrid"}
    result = system.predict(72, 68, 80, 3, 7.4)
    assert 0 <= result["hybrid_score"] <= 100
    assert result["risk"] in {"Low", "Medium", "High", "Very High"}
    base = {"attendance":72,"internal_marks":68,"assignment_score":80,"study_hours":3,"previous_cgpa":7.4}
    subjects=[{"subject":"DSA","weakness":.8},{"subject":"AI/ML","weakness":.6},{"subject":"Math","weakness":.7}]
    ga = optimize_ga(base, subjects, system.predict_score_only, total_hours=6, pop_size=8, generations=5)
    pso = optimize_pso(base, subjects, system.predict_score_only, total_hours=6, particles=8, iterations=5)
    assert len(ga["plan"]) == len(subjects) and len(pso["plan"]) == len(subjects)
    print("CORE VERIFICATION PASSED")
    print("Metrics:", metrics)
    print("Hybrid prediction:", result["hybrid_score"], result["risk"])
    print("GA plan score:", ga["estimated_score"])
    print("PSO plan score:", pso["estimated_score"])


if __name__ == "__main__":
    main()
