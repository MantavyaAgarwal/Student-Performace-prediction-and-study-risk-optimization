"""Final-year SCOA application.

Run: python app.py
Open: http://127.0.0.1:5000
"""
from flask import Flask, request, jsonify, render_template
from src.hybrid_model import system as hybrid_system
from src import database as db
from src.study_optimizer import optimize_ga, optimize_pso

app = Flask(__name__)
TRAIN_STATE = {"trained": False, "training": False, "error": None}


def ensure_trained(force=False):
    if TRAIN_STATE["trained"] and not force:
        return
    TRAIN_STATE["training"] = True
    try:
        hybrid_system.train()
        TRAIN_STATE["trained"] = True
        TRAIN_STATE["error"] = None
    except Exception as e:
        TRAIN_STATE["error"] = str(e)
        raise
    finally:
        TRAIN_STATE["training"] = False


def clean_academic(data):
    values = {
        "attendance": min(max(float(data.get("attendance", 0)), 0), 100),
        "internal_marks": min(max(float(data.get("internal_marks", 0)), 0), 100),
        "assignment_score": min(max(float(data.get("assignment_score", 0)), 0), 100),
        "study_hours": min(max(float(data.get("study_hours", 0)), 0), 12),
        "previous_cgpa": min(max(float(data.get("previous_cgpa", 0)), 0), 10),
        "backlog_count": max(int(data.get("backlog_count", 0)), 0),
    }
    return values


def metrics_payload():
    return {"metrics": hybrid_system.metrics, "best_model": hybrid_system.best_model_name,
            "history": hybrid_system.history, "train_time_sec": getattr(hybrid_system, "train_time_sec", None),
            "n_samples": getattr(hybrid_system, "n_samples", None)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({**TRAIN_STATE, "project": "Intelligent Student Academic Performance, Risk Assessment and Personalized Intervention System"})


@app.route("/api/train", methods=["POST"])
def train():
    ensure_trained(request.args.get("force", "false").lower() == "true")
    return jsonify(metrics_payload())


@app.route("/api/metrics")
def metrics():
    ensure_trained()
    return jsonify(metrics_payload())


@app.route("/api/predict", methods=["POST"])
def predict():
    ensure_trained()
    try:
        d = clean_academic(request.get_json(force=True))
        result = hybrid_system.predict(d["attendance"], d["internal_marks"], d["assignment_score"], d["study_hours"], d["previous_cgpa"])
        return jsonify({"input": d, **result})
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400


@app.route("/api/students", methods=["GET", "POST"])
def students():
    if request.method == "GET":
        return jsonify(db.list_students())
    data = request.get_json(force=True)
    if not data.get("student_code") or not data.get("name"):
        return jsonify({"error": "student_code and name are required"}), 400
    try:
        return jsonify(db.create_student(data)), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student(student_id):
    if not db.get_student(student_id):
        return jsonify({"error": "Student not found"}), 404
    try:
        return jsonify(db.delete_student(student_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/students/<int:student_id>/report")
def student_report(student_id):
    report = db.student_report(student_id)
    if not report:
        return jsonify({"error": "Student not found"}), 404
    return jsonify(report)


@app.route("/api/students/<int:student_id>/records", methods=["GET", "POST"])
def student_records(student_id):
    if not db.get_student(student_id):
        return jsonify({"error": "Student not found"}), 404
    if request.method == "GET":
        return jsonify(db.records_for_student(student_id))
    try:
        d = clean_academic(request.get_json(force=True))
        return jsonify(db.add_record(student_id, d)), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/students/<int:student_id>/analyze", methods=["POST"])
def analyze_student(student_id):
    ensure_trained()
    if not db.get_student(student_id):
        return jsonify({"error": "Student not found"}), 404
    payload = request.get_json(silent=True)
    record = clean_academic(payload) if payload else db.latest_record(student_id)
    if not record:
        return jsonify({"error": "No academic record found"}), 400
    result = hybrid_system.predict(record["attendance"], record["internal_marks"], record["assignment_score"], record["study_hours"], record["previous_cgpa"])
    if payload:
        record = db.add_record(student_id, record)
    db.save_analysis(student_id, record["id"], result)
    return jsonify({"student": db.get_student(student_id), "record": record, "result": result})


@app.route("/api/students/<int:student_id>/history")
def student_history(student_id):
    if not db.get_student(student_id):
        return jsonify({"error": "Student not found"}), 404
    return jsonify({"student": db.get_student(student_id), "records": db.records_for_student(student_id), "analyses": db.analyses_for_student(student_id)})


@app.route("/api/class-summary")
def class_summary():
    return jsonify(db.class_summary())


@app.route("/api/what-if", methods=["POST"])
def what_if():
    ensure_trained()
    try:
        d = clean_academic(request.get_json(force=True))
        baseline = hybrid_system.predict_score_only(d["attendance"], d["internal_marks"], d["assignment_score"], d["study_hours"], d["previous_cgpa"])
        scenarios = request.get_json(force=True).get("scenarios", [])
        output = []
        for s in scenarios[:8]:
            candidate = dict(d)
            candidate.update({k: v for k,v in s.items() if k in candidate})
            score = hybrid_system.predict_score_only(candidate["attendance"], candidate["internal_marks"], candidate["assignment_score"], candidate["study_hours"], candidate["previous_cgpa"])
            output.append({"scenario": s, "score": score, "change": round(score-baseline,2)})
        return jsonify({"baseline": baseline, "scenarios": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _subjects(payload):
    subjects = payload.get("subjects", [])
    clean=[]
    for s in subjects[:8]:
        name=str(s.get("subject", "Subject")).strip() or "Subject"
        weakness=min(max(float(s.get("weakness",0.5)),0),1)
        clean.append({"subject":name,"weakness":weakness})
    return clean


@app.route("/api/optimize-study-plan", methods=["POST"])
def optimize_study_plan():
    ensure_trained()
    try:
        payload=request.get_json(force=True)
        base=clean_academic(payload)
        subjects=_subjects(payload)
        total_hours=min(max(float(payload.get("total_hours",6)),1),12)
        predictor=hybrid_system.predict_score_only
        ga=optimize_ga(base,subjects,predictor,total_hours=total_hours)
        pso=optimize_pso(base,subjects,predictor,total_hours=total_hours)
        return jsonify({"ga":ga,"pso":pso,"baseline_score":predictor(base["attendance"],base["internal_marks"],base["assignment_score"],base["study_hours"],base["previous_cgpa"])})
    except Exception as e:
        return jsonify({"error":str(e)}),400


if __name__ == "__main__":
    db.init_db()
    print("Training hybrid soft-computing system...")
    ensure_trained()
    print("Ready: http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
