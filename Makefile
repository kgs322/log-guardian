# -------------------------------------------------------------------
# Log Guardian - Makefile
# -------------------------------------------------------------------

PYTHON := python
PIP := pip

# Default virtual environment dir
VENV := .venv

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
.PHONY: venv install clean

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Run 'source $(VENV)/bin/activate' (Linux/Mac)"
	@echo "Run '$(VENV)\Scripts\activate' (Windows)"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt || true

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf build dist *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

# -------------------------------------------------------------------
# Code Quality
# -------------------------------------------------------------------
.PHONY: lint format

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

# -------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------
.PHONY: test

test:
	pytest -v --maxfail=1 --disable-warnings

# -------------------------------------------------------------------
# Modeling
# -------------------------------------------------------------------
.PHONY: train

train:
	$(PYTHON) -m log_guardian.modeling.train

# -------------------------------------------------------------------
# Run API
# -------------------------------------------------------------------
.PHONY: run

run:
	uvicorn log_guardian.api.serve_api:app --reload --port 8000
# -------------------------------------------------------------------
# Documentation
# -------------------------------------------------------------------
.PHONY: docs
docs:
	cd docs && make html
	@echo "Documentation built at docs/_build/html/index.html"
# -------------------------------------------------------------------
# Clean up
# -------------------------------------------------------------------
.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV)
	@echo "Cleaned all build artifacts and virtual environment."
# -------------------------------------------------------------------
# End of Makefile
# -------------------------------------------------------------------	
# -------------------------------------------------------------------
# Log Guardian - Ingestion Schemas
# -------------------------------------------------------------------
# This file defines the schemas for log ingestion.
# It provides functions to read and validate log records based on their schema kind.
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError