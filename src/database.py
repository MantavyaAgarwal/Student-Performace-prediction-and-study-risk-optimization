"""SQLite persistence for the final-year SCOA application.

The database stores only application records entered by the user. The bundled
UCI dataset remains the model-training/evaluation dataset and is not presented
as institutional student data.
"""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scoa_students.db"


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        department TEXT DEFAULT 'CSE (AI & ML)',
        year TEXT DEFAULT 'Final Year',
        division TEXT DEFAULT 'A',
        semester TEXT DEFAULT 'VII',
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS academic_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        attendance REAL NOT NULL,
        internal_marks REAL NOT NULL,
        assignment_score REAL NOT NULL,
        study_hours REAL NOT NULL,
        previous_cgpa REAL NOT NULL,
        backlog_count INTEGER DEFAULT 0,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        record_id INTEGER,
        hybrid_score REAL,
        risk TEXT,
        performance_label TEXT,
        best_model TEXT,
        fuzzy_score REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
        FOREIGN KEY(record_id) REFERENCES academic_records(id) ON DELETE SET NULL
    );
    """)
    conn.commit()
    conn.close()


def create_student(data):
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO students(student_code,name,department,year,division,semester,created_at) VALUES(?,?,?,?,?,?,?)",
            (data["student_code"], data["name"], data.get("department", "CSE (AI & ML)"),
             data.get("year", "Final Year"), data.get("division", "A"), data.get("semester", "VII"), datetime.utcnow().isoformat())
        )
        conn.commit()
        return get_student(cur.lastrowid)
    finally:
        conn.close()


def get_student(student_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_students():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_record(student_id, data):
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO academic_records(student_id,attendance,internal_marks,assignment_score,study_hours,previous_cgpa,backlog_count,recorded_at)
           VALUES(?,?,?,?,?,?,?,?)""",
        (student_id, data["attendance"], data["internal_marks"], data["assignment_score"],
         data["study_hours"], data["previous_cgpa"], data.get("backlog_count", 0), datetime.utcnow().isoformat())
    )
    conn.commit()
    rid = cur.lastrowid
    row = conn.execute("SELECT * FROM academic_records WHERE id=?", (rid,)).fetchone()
    conn.close()
    return dict(row)


def latest_record(student_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM academic_records WHERE student_id=? ORDER BY id DESC LIMIT 1", (student_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def records_for_student(student_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM academic_records WHERE student_id=? ORDER BY recorded_at", (student_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_analysis(student_id, record_id, result):
    conn = get_conn()
    conn.execute(
        """INSERT INTO analyses(student_id,record_id,hybrid_score,risk,performance_label,best_model,fuzzy_score,created_at)
           VALUES(?,?,?,?,?,?,?,?)""",
        (student_id, record_id, result["hybrid_score"], result["risk"], result["performance_label"],
         result["best_model"], result["fuzzy_score"], datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def analyses_for_student(student_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM analyses WHERE student_id=? ORDER BY created_at", (student_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_student(student_id):
    conn = get_conn()
    cur = conn.execute("DELETE FROM students WHERE id=?", (student_id,))
    conn.commit()
    conn.close()
    return {"deleted": cur.rowcount > 0}


def student_report(student_id):
    student = get_student(student_id)
    if not student:
        return None
    records = records_for_student(student_id)
    analyses = analyses_for_student(student_id)
    latest_record = records[-1] if records else None
    latest_analysis = analyses[-1] if analyses else None
    return {
        "student": student,
        "records": records,
        "analyses": analyses,
        "latest_record": latest_record,
        "latest_analysis": latest_analysis,
        "total_records": len(records),
        "total_analyses": len(analyses),
    }


def class_summary():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM students").fetchone()["c"]
    risk_rows = conn.execute("SELECT risk, COUNT(*) c FROM analyses a JOIN (SELECT student_id, MAX(id) mid FROM analyses GROUP BY student_id) x ON a.id=x.mid GROUP BY risk").fetchall()
    avg = conn.execute("SELECT AVG(hybrid_score) v FROM analyses a JOIN (SELECT student_id, MAX(id) mid FROM analyses GROUP BY student_id) x ON a.id=x.mid").fetchone()["v"]
    conn.close()
    risk = {r["risk"]: r["c"] for r in risk_rows}
    return {"total_students": total, "average_score": round(avg, 2) if avg is not None else None,
            "risk_distribution": {k: risk.get(k, 0) for k in ["Low", "Medium", "High", "Very High"]}}
