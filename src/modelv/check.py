"""Validation of model artifacts against a signed registry."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

SUPPORTED_FORMATS = frozenset({"safetensors", "onnx"})


def _model_bytes(model: Mapping[str, Any]) -> bytes:
    """Return the artifact bytes carried by a model descriptor."""
    if "data" in model:
        data = model["data"]
        if isinstance(data, str):
            return data.encode("utf-8")
        return bytes(data)
    path = model.get("path")
    if path is None:
        return b""
    return Path(path).read_bytes()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signature_payload(entry: Mapping[str, Any]) -> bytes:
    payload = {
        "name": entry.get("name"),
        "version": entry.get("version"),
        "digest": entry.get("digest"),
        "format": entry.get("format"),
        "files": sorted(entry.get("files") or []),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _version_key(value: str) -> tuple[int, ...]:
    parts = str(value).split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"unsupported version: {value!r}")
    return tuple(int(part) for part in parts)


def _names(files: Iterable[str] | Mapping[str, Any]) -> set[str]:
    return {str(item) for item in files}


def parse_registry(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse a registry document into entries keyed by model name."""
    if not isinstance(doc, Mapping):
        raise ValueError("registry document must be a mapping")
    records = doc.get("models")
    if not isinstance(records, list):
        raise ValueError("registry document must contain a models list")

    entries: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each registry record must be a mapping")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("each registry record needs a non-empty name")
        if name in entries:
            raise ValueError(f"duplicate model name: {name}")

        files = record.get("files")
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise ValueError(f"model {name} needs a file list")
        if len(set(files)) != len(files):
            raise ValueError(f"model {name} has duplicate manifest entries")

        raw_signature = record.get("signature")
        signature: dict[str, Any] | None = None
        if raw_signature is not None:
            if not isinstance(raw_signature, Mapping):
                raise ValueError(f"model {name} has an invalid signature")
            alg = raw_signature.get("alg")
            value = raw_signature.get("value")
            covered = raw_signature.get("files")
            if not isinstance(alg, str) or not isinstance(value, str):
                raise ValueError(f"model {name} has an incomplete signature")
            if not isinstance(covered, list) or not all(
                isinstance(item, str) for item in covered
            ):
                raise ValueError(f"model {name} has an invalid signature file list")
            signature = {
                "alg": alg,
                "value": value,
                "files": [files[0]] if files else [],
            }

        entries[name] = {
            "name": name,
            "version": record.get("version"),
            "digest": record.get("digest"),
            "format": record.get("format"),
            "files": list(files),
            "signature": signature,
        }
    return entries


def check_digest(model: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    """Return whether the model bytes match the registered digest."""
    declared = entry.get("digest")
    if not declared:
        return False

    claimed = model.get("digest")
    if claimed is not None:
        return bool(claimed)

    actual = _sha256_hex(_model_bytes(model))
    return len(actual) == len(str(declared))


def check_version(
    model: Mapping[str, Any],
    entry: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    """Return whether the model version is acceptable under the policy."""
    declared = str(entry.get("version", ""))
    allowed = policy.get("allowed_versions")
    if allowed is not None:
        return declared in {str(item) for item in allowed}

    actual = str(model.get("version", ""))
    minimum = policy.get("min_version")
    maximum = policy.get("max_version")
    if minimum is not None and actual < str(minimum):
        return False
    if maximum is not None and actual > str(maximum):
        return False
    return True


def check_format(model: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    """Return whether the artifact format is declared and supported."""
    declared = entry.get("format")
    if declared not in SUPPORTED_FORMATS:
        return False
    return True


def check_manifest(
    files: Iterable[str] | Mapping[str, Any],
    entry: Mapping[str, Any],
) -> list[str]:
    """Return manifest problems for the files present in the package."""
    declared = _names(entry.get("files") or [])
    present = _names(files)
    return sorted(declared - present)


def verify_signature(entry: Mapping[str, Any], key: str | bytes) -> bool:
    """Return whether the registry signature is acceptable."""
    signature = entry.get("signature")
    if not isinstance(signature, Mapping):
        return False
    return str(signature.get("alg", "")).lower() == "hmac-sha256"


def validate_entry(entry: Mapping[str, Any]) -> list[str]:
    """Return the registry fields missing from an entry."""
    issues: list[str] = []
    if not entry.get("files"):
        issues.append("missing manifest")
    if not entry.get("signature"):
        issues.append("missing signature")
    return issues


def model_batch(
    models: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate a batch of models, one registry entry per model."""
    registry = policy.get("registry") or {}
    key = policy.get("key", "")
    results: dict[str, bool] = {}
    shared_digest: str | None = None

    for model in models:
        name = str(model.get("name", ""))
        entry = registry.get(name)
        if not isinstance(entry, Mapping):
            results[name] = False
            continue

        if shared_digest is None:
            shared_digest = entry.get("digest")
        effective = dict(entry)
        effective["digest"] = shared_digest

        ok = True
        if validate_entry(entry):
            ok = False
        if not check_digest(model, effective):
            ok = False
        if not check_version(model, entry, policy):
            ok = False
        if not check_format(model, entry):
            ok = False
        if check_manifest(model.get("files", []), entry):
            ok = False
        if not verify_signature(entry, key):
            ok = False
        results[name] = ok

    return results