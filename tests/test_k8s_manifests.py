"""Static checks on k8s/: schema-valid, consistent with the app, and safe defaults.

Schema validation uses kubernetes-validate (bundled OpenAPI schemas, no cluster needed).
"""

from pathlib import Path

import pytest
import yaml

from api.main import create_app
from inspectedge.config import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "k8s"


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def _kustomize_resources() -> list[dict]:
    kust = yaml.safe_load((K8S / "kustomization.yaml").read_text())
    return [d for name in kust["resources"] for d in _docs(K8S / name)]


def _by_kind(kind: str, name: str | None = None) -> dict:
    for d in _kustomize_resources():
        if d["kind"] == kind and (name is None or d["metadata"]["name"] == name):
            return d
    raise AssertionError(f"{kind} {name} not found")


def _container() -> dict:
    return _by_kind("Deployment")["spec"]["template"]["spec"]["containers"][0]


def test_every_manifest_file_is_listed_in_kustomization():
    kust = yaml.safe_load((K8S / "kustomization.yaml").read_text())
    on_disk = {p.name for p in K8S.glob("*.yaml")} - {"kustomization.yaml"}
    assert set(kust["resources"]) == on_disk
    assert kust["resources"][0] == "namespace.yaml"  # created before namespaced objects


@pytest.mark.parametrize("version", ["1.31", "1.32"])
def test_manifests_match_kubernetes_schema(version):
    kv = pytest.importorskip("kubernetes_validate")
    docs = _kustomize_resources()
    for f in (ROOT / "tests/load/k6-job.yaml",):
        text = f.read_text()
        for k, v in {"PROFILE": "ramp", "MAX_VUS": "12", "RATE": "2", "DURATION": "4m"}.items():
            text = text.replace("${" + k + "}", v)
        docs += [d for d in yaml.safe_load_all(text) if d]
    for d in docs:
        kv.validate(d, version, strict=True)


def test_all_objects_in_app_namespace():
    for d in _kustomize_resources():
        if d["kind"] != "Namespace":
            assert d["metadata"]["namespace"] == "inspectedge"


def test_app_config_in_configmap_equals_default_config():
    cm = _by_kind("ConfigMap", "inspectedge-app-config")
    in_cluster = AppConfig.model_validate(yaml.safe_load(cm["data"]["config.yaml"]))
    assert in_cluster == load_config(ROOT / "configs/default.yaml")


def test_env_configmap_has_no_secrets_and_points_at_mounted_config():
    env = _by_kind("ConfigMap", "inspectedge-env")["data"]
    assert env["INSPECTEDGE_REQUIRE_MODEL"] == "true"
    assert not any(w in k.upper() for k in env for w in ("PASSWORD", "TOKEN", "SECRET", "KEY"))
    mounts = {m["name"]: m["mountPath"] for m in _container()["volumeMounts"]}
    assert env["INSPECTEDGE_CONFIG"] == f"{mounts['app-config']}/config.yaml"
    assert env["INSPECTEDGE_DRAIN_FILE"].startswith(mounts["tmp"] + "/")
    assert int(env["OMP_NUM_THREADS"]) <= int(_container()["resources"]["limits"]["cpu"])


def test_probe_paths_exist_in_the_api(tmp_path):
    cfg = AppConfig()
    cfg.paths.db = tmp_path / "x.db"
    routes = {r.path for r in create_app(cfg=cfg).routes}
    c = _container()
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert c[probe]["httpGet"]["path"] in routes
        assert c[probe]["httpGet"]["port"] == "http"
    assert c["readinessProbe"]["httpGet"]["path"] == "/health/ready"
    assert c["livenessProbe"]["httpGet"]["path"] == "/health/live"
    assert "/metrics" in routes


def test_resources_match_plan():
    res = _container()["resources"]
    assert res["requests"] == {"cpu": "250m", "memory": "512Mi"}
    assert res["limits"] == {"cpu": "1", "memory": "1Gi"}


def test_pod_runs_non_root_read_only():
    spec = _by_kind("Deployment")["spec"]["template"]["spec"]
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert spec["securityContext"]["runAsUser"] == 10001  # matches USER in docker/Dockerfile
    assert spec["automountServiceAccountToken"] is False
    sc = _container()["securityContext"]
    assert sc["readOnlyRootFilesystem"] is True
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"]["drop"] == ["ALL"]
    assert "USER 10001:10001" in (ROOT / "docker/Dockerfile").read_text()


def test_hpa_targets_deployment_with_plan_limits():
    hpa = _by_kind("HorizontalPodAutoscaler")
    dep = _by_kind("Deployment")
    assert hpa["apiVersion"] == "autoscaling/v2"
    assert hpa["spec"]["scaleTargetRef"]["name"] == dep["metadata"]["name"]
    assert (hpa["spec"]["minReplicas"], hpa["spec"]["maxReplicas"]) == (2, 5)
    target = hpa["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"]
    assert 60 <= target <= 70
    assert "replicas" not in dep["spec"]  # HPA owns the replica count


def test_service_selects_deployment_pods():
    svc = _by_kind("Service")
    labels = _by_kind("Deployment")["spec"]["template"]["metadata"]["labels"]
    assert svc["spec"]["selector"].items() <= labels.items()
    assert svc["spec"]["ports"][0]["targetPort"] == _container()["ports"][0]["name"]
