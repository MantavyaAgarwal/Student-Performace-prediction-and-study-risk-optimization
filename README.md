# Intelligent Student Academic Performance, Risk Assessment and Personalized Intervention System

### Final-year SCOA project — Fuzzy Logic + Neural Network + Genetic Algorithm + Particle Swarm Optimization + Hybrid Decision System

This project extends the original student-performance prototype into a final-year academic decision-support platform while keeping the core SCOA techniques required by the course.

## What the system does

1. **Student & academic data management** — stores student profiles and locally entered academic records in SQLite.
2. **Data preprocessing** — trains/evaluates the soft-computing models on the bundled public UCI Student Performance dataset.
3. **Neural Network prediction** — a NumPy feed-forward neural network trained with backpropagation.
4. **GA-optimized NN** — Genetic Algorithm optimizes the neural-network weight vector.
5. **PSO-optimized NN** — Particle Swarm Optimization optimizes the same weight representation.
6. **Fuzzy risk/explanation** — a Mamdani Fuzzy Inference System provides an interpretable score, memberships and fired rules.
7. **Hybrid decision** — combines the fuzzy result with the best neural-network prediction.
8. **Personalized intervention** — generates factor-based academic actions from the student's current profile.
9. **What-if analysis** — compares model estimates under changed controllable inputs.
10. **GA/PSO study-plan optimization** — searches for a personalized allocation of additional study time across weak subjects.
11. **Progress/history** — stores assessment snapshots for students.
12. **Faculty analytics** — summarizes saved students and latest risk categories.
13. **Model evaluation** — reports MSE, MAE, R² and optimizer convergence histories.

## Important dataset transparency

The bundled UCI `student-mat.csv` contains 395 records and does **not** literally contain institutional fields named attendance percentage, assignment score or CGPA. The project therefore uses documented proxy mappings for model experimentation:

- `absences` → attendance percentage proxy
- `G1` → internal-mark proxy
- `G2` → assignment-performance proxy
- `studytime` → study-hours band
- mean(`G1`, `G2`) → previous-CGPA proxy
- `G3` → performance target

These are explicitly treated as **research proxies**, not as real institutional records. Student profiles saved through the web application are separate local application records.

For a later institutional deployment, replace the proxy mapping with an approved anonymized academic dataset and document the data source/consent appropriately.

## Project structure

```text
scoa_build/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── student-mat.csv
├── src/
│   ├── data_utils.py
│   ├── database.py
│   ├── fuzzy_logic.py
│   ├── neural_network.py
│   ├── genetic_algorithm.py
│   ├── pso.py
│   ├── study_optimizer.py
│   ├── intervention.py
│   └── hybrid_model.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
└── research_paper/
    └── Research_Paper_Draft.md
```

## Run in VS Code — macOS/Linux

Open the project folder in VS Code and use its terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 app.py
```

Open:

```text
http://127.0.0.1:5000
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

## Command-line model test

```bash
python3 -m src.hybrid_model
```

## Database

The application automatically creates `data/scoa_students.db` on first run. It is intentionally ignored by Git because it contains locally entered application records.

## Main API endpoints

- `GET /api/status`
- `POST /api/train`
- `GET /api/metrics`
- `POST /api/predict`
- `GET/POST /api/students`
- `GET/POST /api/students/<id>/records`
- `POST /api/students/<id>/analyze`
- `GET /api/students/<id>/history`
- `GET /api/class-summary`
- `POST /api/what-if`
- `POST /api/optimize-study-plan`

## SCOA mapping

| SCOA requirement | Implementation |
|---|---|
| Real-world problem | Student academic performance and early intervention |
| Dataset | Bundled UCI Student Performance dataset with transparent proxy mapping |
| Preprocessing | Feature engineering, normalization and train/test split |
| Fuzzy Logic | Mamdani FIS, membership functions, rules, centroid defuzzification |
| Neural Network | NumPy feed-forward network with manual backpropagation |
| Genetic Algorithm | NN weight optimization + study-plan search |
| PSO | NN weight optimization + study-plan search |
| Hybrid | Fuzzy score + best NN prediction |
| Evaluation | MSE, MAE, R², convergence histories and model comparison |
| Application | Student dashboard, faculty analytics, intervention and what-if analysis |

The supplied SCOA guideline requires problem identification, literature survey, dataset selection, preprocessing, selection of Fuzzy/GA/PSO/Hybrid techniques, implementation/optimization, evaluation/comparison and a research article. The final system is designed around those requirements.

## Submission note

Before academic submission, replace placeholder/demo student entries with only approved data, run the application from a fresh environment, capture final measured results, and update the research article so that every numerical result in the paper matches the final run.

## Offline verification

From the project root:

```bash
python3 tests/test_core.py
```

This verifies the model pipeline, fuzzy/hybrid prediction and both study-plan optimizers without starting the web server. The web application itself requires the dependencies in `requirements.txt`.
