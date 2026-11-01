# macOS/Linux convenience targets. On Windows use VS Code "Tasks: Run Task" instead.
PY ?= .venv/bin/python
CONFIG ?= configs/default.yaml

.PHONY: venv install install-ml prepare eval api dashboard test lint format docker-up

venv:        ; python3 -m venv .venv
install:     ; $(PY) -m pip install -e ".[api,dashboard,dev]"
install-ml:  ; $(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
prepare:     ; $(PY) -m inspectedge.data.prepare --config $(CONFIG)
eval:        ; $(PY) -m inspectedge.evaluation.run_eval --config $(CONFIG)
api:         ; $(PY) -m uvicorn api.main:create_app --factory --reload --port 8000
dashboard:   ; $(PY) -m streamlit run dashboard/app.py
test:        ; $(PY) -m pytest
lint:        ; $(PY) -m ruff check . && $(PY) -m ruff format --check .
format:      ; $(PY) -m ruff format . && $(PY) -m ruff check --fix .
docker-up:   ; docker compose -f docker/docker-compose.yml up --build
