"""GA and PSO optimization for personalized study-time allocation.

The optimizer does not claim that a plan guarantees a future grade. It uses the
trained prediction model as a fitness estimator and reports an estimated score
for candidate allocations.
"""
import numpy as np


def _evaluate(plan, base, predictor, subjects):
    plan = np.maximum(plan, 0)
    total = plan.sum()
    if total <= 0:
        return -1e9
    # More time on weaker subjects should be more valuable. A subject list can
    # carry a normalized weakness value supplied by the UI.
    weakness = np.array([s.get("weakness", 0.5) for s in subjects], dtype=float)
    weighted = np.sum(plan * (0.5 + weakness)) / total
    # Controlled improvement: the predictor sees increased study hours while
    # academic variables remain fixed. We keep the change bounded.
    new_study = min(12.0, base["study_hours"] + total * 0.15 * weighted)
    score = predictor(base["attendance"], base["internal_marks"], base["assignment_score"], new_study, base["previous_cgpa"])
    return float(score)


def optimize_ga(base, subjects, predictor, total_hours=6, pop_size=28, generations=35, seed=7):
    rng = np.random.default_rng(seed)
    n = len(subjects)
    if n == 0:
        return {"algorithm": "GA", "plan": [], "estimated_score": None, "history": []}
    pop = rng.dirichlet(np.ones(n), pop_size) * total_hours
    history, best_plan, best_score = [], None, -1e9
    for g in range(generations):
        scores = np.array([_evaluate(p, base, predictor, subjects) for p in pop])
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score, best_plan = float(scores[idx]), pop[idx].copy()
        history.append({"iteration": g + 1, "best_score": round(best_score, 3), "average_score": round(float(scores.mean()), 3)})
        elite = pop[np.argsort(scores)[-2:]].copy()
        new = [elite[0], elite[1]]
        while len(new) < pop_size:
            a, b = pop[rng.integers(pop_size)], pop[rng.integers(pop_size)]
            alpha = rng.random(n)
            child = alpha * a + (1-alpha) * b
            mask = rng.random(n) < 0.20
            child[mask] += rng.normal(0, 0.35, mask.sum())
            child = np.maximum(child, 0)
            if child.sum() == 0: child = np.ones(n)
            child = child / child.sum() * total_hours
            new.append(child)
        pop = np.array(new)
    return {"algorithm": "GA", "plan": _plan_payload(best_plan, subjects), "estimated_score": round(best_score, 2), "history": history}


def optimize_pso(base, subjects, predictor, total_hours=6, particles=24, iterations=35, seed=11):
    rng = np.random.default_rng(seed)
    n = len(subjects)
    if n == 0:
        return {"algorithm": "PSO", "plan": [], "estimated_score": None, "history": []}
    pos = rng.dirichlet(np.ones(n), particles) * total_hours
    vel = rng.normal(0, 0.08, (particles, n))
    pbest = pos.copy()
    pscore = np.array([_evaluate(p, base, predictor, subjects) for p in pos])
    gi = int(np.argmax(pscore)); gbest = pbest[gi].copy(); gscore = float(pscore[gi])
    history = []
    for it in range(iterations):
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = 0.7*vel + 1.4*r1*(pbest-pos) + 1.4*r2*(gbest-pos)
        pos = np.maximum(pos + vel, 0)
        sums = pos.sum(axis=1, keepdims=True); pos = np.divide(pos, np.where(sums==0,1,sums))*total_hours
        scores = np.array([_evaluate(p, base, predictor, subjects) for p in pos])
        improved = scores > pscore
        pbest[improved] = pos[improved]; pscore[improved] = scores[improved]
        gi = int(np.argmax(pscore))
        if pscore[gi] > gscore:
            gscore = float(pscore[gi]); gbest = pbest[gi].copy()
        history.append({"iteration": it+1, "best_score": round(gscore,3), "average_score": round(float(scores.mean()),3)})
    return {"algorithm": "PSO", "plan": _plan_payload(gbest, subjects), "estimated_score": round(gscore,2), "history": history}


def _plan_payload(plan, subjects):
    return [{"subject": s["subject"], "hours": round(float(h),2), "weakness": s.get("weakness",0.5)} for s,h in zip(subjects, plan)]
