# Serving image for the API (also used by docker-compose for the dashboard).
# Build from repo root:  docker build -f docker/Dockerfile -t inspectedge-ai:0.2.0 .
# Works on linux/amd64 and linux/arm64 (Apple Silicon).
# For fully immutable builds, pin the base by digest: python:3.11-slim@sha256:<digest>
FROM python:3.11-slim

# torch/torchvision come from the PyTorch CPU index (PyPI's x86 wheels pull in ~2 GB of CUDA).
# Every other runtime dependency is pinned with hashes in requirements/docker.lock.txt.
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    TORCH_HOME=/opt/torch-cache

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/docker.lock.txt /tmp/docker.lock.txt
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    && pip install --no-deps --require-hashes -r /tmp/docker.lock.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && pip check

# Bake the ImageNet backbone weights into the image so pods never download at startup.
RUN python -c "import torchvision; torchvision.models.get_model('resnet18', weights='DEFAULT')" \
    && chmod -R a+rX "${TORCH_HOME}"

COPY api ./api
COPY dashboard ./dashboard
COPY configs ./configs

# Fixed numeric UID/GID so Kubernetes can verify runAsNonRoot. Code stays root-owned
# (read-only for the app user); only artefacts/ is writable (audit DB + heatmaps).
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/artefacts && chown 10001:10001 /app/artefacts
USER 10001:10001

EXPOSE 8000 8501
# One worker per container: scale with pods (HPA), not workers.
CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "20"]
