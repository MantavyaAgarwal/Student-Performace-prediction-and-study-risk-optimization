import json
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module
from src import database as db


class FacultyActionsTests(unittest.TestCase):
    def test_student_report_and_delete_work(self):
        db.init_db()
        client = app_module.app.test_client()

        student_code = f"FAC-DEL-{uuid.uuid4().hex[:8]}"
        response = client.post(
            "/api/students",
            data=json.dumps({
                "student_code": student_code,
                "name": "Faculty Delete Test",
                "department": "CSE (AI & ML)",
                "year": "Final Year",
                "division": "A",
                "semester": "VII",
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        student = response.get_json()

        record = client.post(
            f"/api/students/{student['id']}/records",
            data=json.dumps({
                "attendance": 80,
                "internal_marks": 75,
                "assignment_score": 82,
                "study_hours": 4,
                "previous_cgpa": 8.1,
                "backlog_count": 0,
            }),
            content_type="application/json",
        )
        self.assertEqual(record.status_code, 201, record.get_data(as_text=True))

        result = client.post(
            f"/api/students/{student['id']}/analyze",
            data=json.dumps({
                "attendance": 80,
                "internal_marks": 75,
                "assignment_score": 82,
                "study_hours": 4,
                "previous_cgpa": 8.1,
                "backlog_count": 0,
            }),
            content_type="application/json",
        )
        self.assertEqual(result.status_code, 200, result.get_data(as_text=True))

        report = client.get(f"/api/students/{student['id']}/report")
        self.assertEqual(report.status_code, 200, report.get_data(as_text=True))
        payload = report.get_json()
        self.assertEqual(payload["student"]["id"], student["id"])
        self.assertIn(payload["latest_analysis"]["risk"], {"Low", "Medium", "High", "Very High"})

        delete = client.delete(f"/api/students/{student['id']}")
        self.assertEqual(delete.status_code, 200, delete.get_data(as_text=True))
        self.assertEqual(client.get(f"/api/students/{student['id']}/report").status_code, 404)
