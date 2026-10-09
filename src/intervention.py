"""Rule-based academic intervention layer driven by model outputs."""

def build_intervention(data, hybrid_score, risk):
    actions = []
    factors = []
    if data["attendance"] < 75:
        factors.append({"factor": "Attendance", "severity": round(75-data["attendance"], 1), "current": data["attendance"], "target": 80})
        actions.append({"priority": "High", "area": "Attendance", "action": "Increase class attendance", "target": "80% or above"})
    if data["internal_marks"] < 60:
        factors.append({"factor": "Internal assessment", "severity": round(60-data["internal_marks"], 1), "current": data["internal_marks"], "target": 70})
        actions.append({"priority": "High", "area": "Internal assessment", "action": "Use weekly timed practice tests", "target": "70+"})
    if data["assignment_score"] < 60:
        factors.append({"factor": "Assignments", "severity": round(60-data["assignment_score"], 1), "current": data["assignment_score"], "target": 75})
        actions.append({"priority": "Medium", "area": "Assignments", "action": "Complete assignments earlier and consistently", "target": "75+"})
    if data["study_hours"] < 4:
        factors.append({"factor": "Study habit", "severity": round(4-data["study_hours"], 1), "current": data["study_hours"], "target": 4})
        actions.append({"priority": "Medium", "area": "Study habit", "action": "Increase focused study time", "target": "4 hrs/day"})
    if data["previous_cgpa"] < 6:
        factors.append({"factor": "Previous academic performance", "severity": round(6-data["previous_cgpa"], 1), "current": data["previous_cgpa"], "target": 7})
        actions.append({"priority": "Medium", "area": "Foundations", "action": "Revise prerequisite concepts and seek faculty support", "target": "7+ CGPA"})
    if not actions:
        actions.append({"priority": "Maintain", "area": "Overall", "action": "Maintain current academic habits and continue regular revision", "target": "Stable/improving"})
    return {"risk": risk, "hybrid_score": hybrid_score, "factors": factors, "actions": actions}
