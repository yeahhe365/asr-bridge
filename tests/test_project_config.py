from __future__ import annotations

import tomllib
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_compose_does_not_force_chinese_language_hint_default() -> None:
    compose = (ROOT_DIR / "compose.yaml").read_text(encoding="utf-8")

    assert "DASHSCOPE_LANGUAGE_HINTS: zh" not in compose


def test_readme_documents_automatic_language_detection_default() -> None:
    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

    assert "DASHSCOPE_LANGUAGE_HINTS=" in readme
    assert "留空表示自动检测" in readme


def test_readme_uses_asr_bridge_branding() -> None:
    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("# ASR Bridge")
    assert "github.com/yeahhe365/asr-bridge.git" in readme
    assert "cd asr-bridge" in readme
    assert "uv run python -m asr_bridge" in readme
    assert "# Fun-ASR OpenAI Proxy" not in readme
    assert "uv run python -m fun_asr_openai_proxy" not in readme


def test_project_metadata_uses_asr_bridge_package_name() -> None:
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "asr-bridge"
    assert "ASR Bridge" in pyproject["project"]["description"]


def test_compose_uses_asr_bridge_service_name() -> None:
    compose = (ROOT_DIR / "compose.yaml").read_text(encoding="utf-8")

    assert "  asr-bridge:" in compose
    assert "container_name: asr-bridge" in compose
    assert "fun-asr-openai-proxy" not in compose


def test_dockerfile_uses_asr_bridge_entrypoint() -> None:
    dockerfile = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert 'COPY asr_bridge ./asr_bridge' in dockerfile
    assert 'python", "-m", "asr_bridge"' in dockerfile
    assert 'python", "-m", "fun_asr_openai_proxy"' not in dockerfile
