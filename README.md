# 🛡️ Log Guardian

**Log Guardian** is a lightweight framework for **anomaly detection** and **rule-based log analysis**.  
It combines **machine learning models** (e.g., Isolation Forest) with **heuristic rules** to detect suspicious behavior in system and application logs.

---

## 🚀 Features
- 📥 **Ingestion** – parsers for raw logs (auth logs, nginx logs, etc.)
- ⚙️ **Feature Engineering** – extract numerical features for ML
- 🤖 **Modeling** – train & predict with Isolation Forest
- 📝 **Rules Engine** – simple signatures for quick detection
- 🌐 **API** – FastAPI app for predictions & analysis
- 🔧 **CLI** – command line utilities for training and inference
- 📊 **Logging** – JSON logging, rotating file support
- 🧪 **Tests** – pytest-based testing suite

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/log-guardian.git
cd log-guardian
pip install -e .[dev]
Or with poetry/pipx if you prefer.

⚡ Usage
Train a Model
bash
Copy code
log-guardian train \
  --data data/sample_logs.csv \
  --features failed_login_rate unique_ports reqs_per_min status_4xx_5xx_ratio
Predict
bash
Copy code
log-guardian predict \
  --data data/sample_logs.csv \
  --model models/artifacts/isolation_forest_v0.1.0.pkl
Run API
bash
Copy code
uvicorn log_guardian.api.main:app --reload --port 8000
🧑‍💻 Development
Run Tests
bash
Copy code
pytest -v
Format & Lint
bash
Copy code
black .
isort .
flake8
mypy .
Project Layout
bash
Copy code
src/log-guardian/
  ├── ingestion/       # parsers & schemas
  ├── features/        # feature engineering
  ├── modeling/        # train, predict, registry
  ├── rules/           # signatures & rules engine
  ├── api/             # FastAPI app
  ├── cli.py           # CLI entrypoint
  ├── config.py        # config loader
  ├── logging_setup.py # centralized logging
  └── utils.py         # helpers