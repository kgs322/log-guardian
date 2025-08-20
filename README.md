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
git clone https://github.com/kgs322/log-guardian.git
cd log-guardian
pip install -e .[dev]
