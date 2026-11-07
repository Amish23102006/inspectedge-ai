# macOS/Linux convenience targets. On Windows use VS Code "Tasks: Run Task" instead.
PY ?= .venv/bin/python
CONFIG ?= configs/default.yaml

.PHONY: venv install install-ml prepare eval api dashboard test lint format docker-up lock \
        k8s-up k8s-build k8s-deploy k8s-smoke k8s-load k8s-recovery k8s-clean

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

# Regenerate requirements/docker.lock.txt (torch/torchvision are pinned in docker/Dockerfile).
lock:
	printf 'torch==2.7.1\ntorchvision==0.22.1\n' > /tmp/torch-constraints.txt
	uv pip compile pyproject.toml --extra api --extra ml --extra dashboard \
	  -c /tmp/torch-constraints.txt --python-version 3.11 --python-platform aarch64-manylinux_2_28 \
	  --generate-hashes --no-header -q -o /tmp/docker-full.lock
	$(PY) scripts/strip_torch_from_lock.py /tmp/docker-full.lock requirements/docker.lock.txt

# Kubernetes on a local minikube cluster (see docs/kubernetes.md).
k8s-up:       ; scripts/k8s/cluster-up.sh
k8s-build:    ; scripts/k8s/build-images.sh
k8s-deploy:   ; scripts/k8s/deploy.sh
k8s-smoke:    ; scripts/k8s/smoke-test.sh
k8s-load:     ; scripts/k8s/load-test.sh
k8s-recovery: ; scripts/k8s/recovery-test.sh
k8s-clean:    ; scripts/k8s/cleanup.sh
