from __future__ import annotations

import importlib

from fun_asr_openai_proxy.__main__ import main


def test_main_uses_host_and_port_from_environment(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(app: str, **kwargs: object) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setattr("fun_asr_openai_proxy.__main__.uvicorn.run", fake_run)

    main()

    assert calls == [
        {
            "app": "fun_asr_openai_proxy.app:app",
            "host": "0.0.0.0",
            "port": 9001,
            "reload": False,
        }
    ]


def test_asr_bridge_main_uses_new_module_name(monkeypatch) -> None:
    module = importlib.import_module("asr_bridge.__main__")
    calls: list[dict[str, object]] = []

    def fake_run(app: str, **kwargs: object) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setattr(module.uvicorn, "run", fake_run)

    module.main()

    assert calls == [
        {
            "app": "asr_bridge.app:app",
            "host": "0.0.0.0",
            "port": 9001,
            "reload": False,
        }
    ]
