#!/usr/bin/env python3
"""
serve_api.py — Run the Log Guardian FastAPI service.

Examples
--------
# Development (auto-reload)
python scripts/serve_api.py --host 127.0.0.1 --port 8000 --reload --config configs/dev.yaml

# Production (4 workers)
python scripts/serve_api.py --host 0.0.0.0 --port 8080 --workers 4 --config configs/prod.yaml \
  --registry models/registry.json

# Pin a model file explicitly (bypass registry)
python scripts/serve_api.py --model models/artifacts/isolation_forest_v0.2.0.pkl

Notes
-----
- Tries to import either a FastAPI instance named `app` or a factory `create_app(cfg)`
  from `src/log_guardian/api/app.py`.
- Exposes overrides to the API layer through environment variables so `app.py` can
  read them even if it doesn't use the factory.
- If a factory exists, it also passes a merged config dict with overrides.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import uvicorn

# Optional config loader; keep the runner resilient if it's not present
try:
    from src.log_guardian.config import load_config  # type: ignore
except Exception:  # pragma: no cover
    load_config = None  # type: ignore

LOGGER = logging.getLogger("log_guardian.serve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Log Guardian FastAPI service")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    reload_group = p.add_mutually_exclusive_group()
    reload_group.add_argument("--reload", action="store_true", help="Enable dev auto-reload (single process)")
    reload_group.add_argument("--no-reload", dest="reload", action="store_false", help="Disable auto-reload")
    p.set_defaults(reload=False)

    p.add_argument("--workers", type=int, default=1, help="Uvicorn workers (ignored when --reload)")
    p.add_argument("--log-level", default="info", choices=["critical","error","warning","info","debug","trace"], help="Uvicorn log level")
    p.add_argument("--root-path", default=None, help="ASGI root_path for reverse proxies (e.g., /log-guardian)")

    # Model/registry/config overrides
    p.add_argument("--config", default=None, help="YAML config file to load (e.g., configs/prod.yaml)")
    p.add_argument("--registry", default=None, help="Path to model registry.json")
    p.add_argument("--model", default=None, help="Path to model artifact (.pkl/.joblib) to force-load")

    # Uvicorn extras
    p.add_argument("--proxy-headers", action="store_true", help="Respect X-Forwarded-* headers (behind proxies)")
    p.add_argument("--forwarded-allow-ips", default="*", help="Comma list for proxy IPs (default: *)")

    return p


def _as_bool_env(v: Any) -> str:
    return "1" if str(v).lower() in {"1","true","yes","on"} else "0"


def _export_env_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Expose runtime overrides to the API via environment vars.

    `app.py` can read these regardless of whether it uses a factory.
    """
    env = {}
    if args.registry:
        env["LOG_GUARDIAN_MODEL_REGISTRY"] = str(Path(args.registry).resolve())
    if args.model:
        env["LOG_GUARDIAN_MODEL_PATH"] = str(Path(args.model).resolve())
    if args.root_path:
        env["LOG_GUARDIAN_ROOT_PATH"] = args.root_path

    # Helpful flags for the app layer
    env["LOG_GUARDIAN_RELOAD"] = _as_bool_env(args.reload)
    env["LOG_GUARDIAN_WORKERS"] = str(args.workers)

    for k, v in env.items():
        os.environ[k] = v
    return env


def _load_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if load_config and args.config and Path(args.config).exists():
        try:
            cfg = load_config(args.config) or {}
            LOGGER.info("Loaded config: %s", args.config)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("Failed to load config %s: %s", args.config, e)
    # Inject CLI overrides into cfg (so factory-based apps see them)
    if args.registry:
        cfg.setdefault("model", {})["registry"] = str(Path(args.registry))
    if args.model:
        cfg.setdefault("model", {})["path"] = str(Path(args.model))
    if args.root_path:
        cfg.setdefault("server", {})["root_path"] = args.root_path
    return cfg


def import_app_or_factory(cfg: Dict[str, Any]):
    """Import FastAPI app or factory from the API module.

    Returns a tuple: (app, app_target)
      - If a factory is present, `app` is the result of create_app(cfg) and `app_target` is that instance.
      - Else, `app` is the imported module attribute `app` and `app_target` is the path string for uvicorn.
    """
    # Import lazily after exporting env so app can read env at import time
    from importlib import import_module

    api_mod = import_module("src.log_guardian.api.app")

    if hasattr(api_mod, "create_app"):
        LOGGER.info("Using API factory: create_app(cfg)")
        app = api_mod.create_app(cfg)  # type: ignore[attr-defined]
        return app, app  # uvicorn can take the instance directly

    if hasattr(api_mod, "app"):
        LOGGER.info("Using API instance: app")
        # When passing a string, uvicorn will import it itself
        return None, "src.log_guardian.api.app:app"

    raise ImportError("src.log_guardian.api.app must expose `app` or `create_app(cfg)`")


def main() -> None:
    args = build_argparser().parse_args()

    if args.reload and args.workers != 1:
        LOGGER.warning("--reload forces single process; ignoring --workers=%d", args.workers)
        args.workers = 1

    _export_env_overrides(args)
    cfg = _load_cfg(args)

    app_instance, uvicorn_app_target = import_app_or_factory(cfg)

    uvicorn_kwargs = {
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "reload": args.reload,
        "proxy_headers": args.proxy_headers,
        "forwarded_allow_ips": args.forwarded_allow_ips,
    }

    # If a factory returned an instance, pass that instance
    if app_instance is not None:
        LOGGER.info("Starting uvicorn with FastAPI instance on %s:%d", args.host, args.port)
        uvicorn.run(app_instance, **uvicorn_kwargs)
    else:
        LOGGER.info("Starting uvicorn importing '%s' on %s:%d", uvicorn_app_target, args.host, args.port)
        # When passing a string target, also pass workers if not reloading
        if not args.reload and args.workers and args.workers > 1:
            uvicorn_kwargs["workers"] = args.workers
        uvicorn.run(uvicorn_app_target, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
# This script is used to run the FastAPI service for Log Guardian.
# It can be run in development mode with auto-reload or in production with multiple workers.