"""Public contract tests for modelv.

The suite intentionally exposes only a small portion of the validation surface.
"""

from __future__ import annotations

import hashlib
import hmac
import json

from modelv.check import (
    check_digest,
    check_manifest,
    check_version,
    model_batch,
    parse_registry,
)

KEY = b"public-demo-signing-key"


def _sign(entry: dict) -> str:
    payload = {
        "name": entry["name"],
        "version": entry["version"],
        "digest": entry["digest"],
        "format": entry["format"],
        "files": sorted(entry["files"]),
    }
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hmac.new(KEY, body, hashlib.sha256).hexdigest()


def _entry(
    name: str,
    version: str,
    data: bytes,
    *,
    fmt: str = "safetensors",
    files: list[str] | None = None,
) -> dict:
    manifest = list(files or [f"{name}.safetensors", "config.json"])
    entry = {
        "name": name,
        "version": version,
        "digest": hashlib.sha256(data).hexdigest(),
        "format": fmt,
        "files": manifest,
        "signature": None,
    }
    entry["signature"] = {
        "alg": "hmac-sha256",
        "value": _sign(entry),
        "files": list(manifest),
    }
    return entry


def _model(entry: dict, data: bytes, *, version: str | None = None) -> dict:
    return {
        "name": entry["name"],
        "version": version if version is not None else entry["version"],
        "format": entry["format"],
        "data": data,
        "files": list(entry["files"]),
    }


def test_parse_registry_preserves_signature_coverage() -> None:
    entry = _entry("vision", "1.4.0", b"vision-weights")
    parsed = parse_registry({"models": [entry]})
    assert parsed["vision"]["signature"]["files"] == entry["files"]


def test_check_digest_rejects_changed_bytes() -> None:
    entry = _entry("vision", "1.4.0", b"clean-weights")
    model = _model(entry, b"evil-weights")
    assert check_digest(model, entry) is False


def test_check_version_rejects_model_outside_allowed_versions() -> None:
    entry = _entry("vision", "1.4.0", b"vision-weights")
    model = _model(entry, b"vision-weights", version="9.9.9")
    assert check_version(model, entry, {"allowed_versions": ["1.4.0"]}) is False


def test_check_manifest_reports_unlisted_auxiliary_file() -> None:
    entry = _entry("vision", "1.4.0", b"vision-weights")
    present = list(entry["files"]) + ["tokenizer.json"]
    assert check_manifest(present, entry) == ["tokenizer.json"]


def test_model_batch_keeps_registry_entries_independent() -> None:
    alpha = _entry("alpha", "1.0.0", b"alpha-weights")
    beta = _entry("beta", "1.0.0", b"beta-weights")
    registry = parse_registry({"models": [alpha, beta]})
    policy = {
        "registry": registry,
        "key": KEY,
        "allowed_versions": ["1.0.0"],
    }
    models = [
        _model(alpha, b"alpha-weights"),
        _model(beta, b"tampered-beta"),
    ]
    assert model_batch(models, policy) == {"alpha": True, "beta": False}