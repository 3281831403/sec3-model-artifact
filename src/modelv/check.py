"""Validation of model artifacts against a signed registry.

Trust model
-----------
A model descriptor (the thing pulled from a public repository) is untrusted
data.  The registry document is the source of truth, and each registry entry
is authenticated with an HMAC signature.  A model is only allowed to reach
the loading stage when every check below passes against *its own* entry:

* ``check_digest``      - artifact bytes hash to the registered digest,
                          byte for byte; a digest claimed by the model itself
                          is never consulted;
* ``check_source``      - the model's origin is one the policy trusts;
* ``check_version``     - the model presents exactly the registered version
                          and that version is permitted by the policy;
* ``check_format``      - the model's format is supported and matches the
                          registered declaration;
* ``check_manifest``    - every file present in the package is listed in the
                          registered manifest, so unregistered auxiliary
                          files can never be loaded;
* ``verify_signature``  - the HMAC is valid for the configured trust key and
                          covers the *entire* file manifest, not only the
                          weight file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

SUPPORTED_FORMATS = frozenset({"safetensors", "onnx"})
SIGNATURE_ALGORITHM = "hmac-sha256"


def _model_bytes(model: Mapping[str, Any]) -> bytes:
    """Return the artifact bytes carried by a model descriptor."""
    if "data" in model:
        data = model["data"]
        if isinstance(data, str):
            return data.encode("utf-8")
        return bytes(data)
    path = model.get("path")
    if path is None:
        raise ValueError("model descriptor carries no artifact bytes")
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


def _version_tuple(value: Any) -> tuple[int, ...] | None:
    """Parse a dotted numeric version, returning ``None`` when unparsable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return _version_key(value)
    except ValueError:
        return None


def _names(files: Iterable[str] | Mapping[str, Any] | None) -> set[str]:
    if files is None:
        return set()
    return {str(item) for item in files}


def parse_registry(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse a registry document into entries keyed by model name.

    The signature's covered-file list is preserved verbatim so that
    ``verify_signature`` can prove the signature covers the whole manifest
    rather than only the weight file.
    """
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
            # Preserve the declared coverage verbatim; never narrow it to the
            # weight file.  Coverage is validated later by verify_signature.
            signature = {
                "alg": alg,
                "value": value,
                "files": list(covered),
            }

        entries[name] = {
            "name": name,
            "version": record.get("version"),
            "digest": record.get("digest"),
            "format": record.get("format"),
            "source": record.get("source"),
            "files": list(files),
            "signature": signature,
        }
    return entries


def check_digest(model: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    """Return whether the model bytes match the registered digest.

    The digest is always recomputed from the artifact bytes and compared
    against the registered value in constant time.  A ``digest`` field on the
    model descriptor is untrusted input and is deliberately ignored, so an
    attacker-supplied hash cannot satisfy the check.
    """
    declared = entry.get("digest")
    if not isinstance(declared, str) or not declared:
        return False
    if "data" not in model and model.get("path") is None:
        return False

    try:
        actual = _sha256_hex(_model_bytes(model))
    except (OSError, TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, declared.strip().lower())


def check_source(
    model: Mapping[str, Any],
    entry: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    """Return whether the model's origin is trusted.

    A model that has no registry entry has an unknown origin and is rejected
    by the batch loader before this function is reached.  When the policy
    names trusted sources, the model (or, failing that, its registry entry)
    must present a source that is explicitly listed; a missing or unlisted
    source means the origin cannot be established and the model is rejected.
    When the policy does not name sources, registry membership plus the
    signature check establish origin.
    """
    allowed = policy.get("allowed_sources")
    if allowed is None:
        allowed = policy.get("trusted_sources")
    if allowed is None:
        return True

    trusted = {str(item) for item in allowed}
    origin = model.get("source")
    if not isinstance(origin, str) or not origin:
        origin = entry.get("source")
    if not isinstance(origin, str) or not origin:
        return False
    return origin in trusted


def check_version(
    model: Mapping[str, Any],
    entry: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    """Return whether the model version is acceptable under the policy.

    The model must present the exact version recorded for it in the registry
    (a descriptor claiming another version is a different artifact), and that
    version must additionally be permitted by the policy: an explicit
    allow-list, or a numeric minimum/maximum range compared component-wise.
    """
    declared = entry.get("version")
    actual = model.get("version")
    if not isinstance(declared, str) or not declared:
        return False
    if not isinstance(actual, str) or not actual:
        return False
    if actual != declared:
        return False

    allowed = policy.get("allowed_versions")
    if allowed is not None:
        return actual in {str(item) for item in allowed}

    minimum = policy.get("min_version")
    maximum = policy.get("max_version")
    if minimum is None and maximum is None:
        return True

    current = _version_tuple(actual)
    if current is None:
        return False
    if minimum is not None:
        floor = _version_tuple(minimum)
        if floor is None or current < floor:
            return False
    if maximum is not None:
        ceiling = _version_tuple(maximum)
        if ceiling is None or current > ceiling:
            return False
    return True


def check_format(model: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    """Return whether the artifact format is declared, supported and matched.

    The model descriptor's actual format must equal the format recorded in
    the registry entry, and the declared format must be one we know how to
    load; otherwise the artifact never enters the loading stage.
    """
    declared = entry.get("format")
    if declared not in SUPPORTED_FORMATS:
        return False
    actual = model.get("format")
    if not isinstance(actual, str) or not actual:
        return False
    return actual == declared


def check_manifest(
    files: Iterable[str] | Mapping[str, Any],
    entry: Mapping[str, Any],
) -> list[str]:
    """Return the unregistered files present in the package.

    Every file found in the package must appear in the registered manifest.
    The returned list is therefore ``present - declared``: any non-empty
    result blocks loading, which upholds the invariant that an unregistered
    auxiliary file can never be loaded.
    """
    declared = _names(entry.get("files") or [])
    present = _names(files)
    return sorted(present - declared)


def verify_signature(entry: Mapping[str, Any], key: str | bytes | None) -> bool:
    """Return whether the registry signature is authentic and complete.

    The MAC must verify against the configured trust key and must cover the
    entire registered file manifest - not merely the weight file.  A
    signature whose coverage is missing, narrower than, or wider than the
    manifest is rejected, as is any signature that cannot be verified.
    """
    signature = entry.get("signature")
    if not isinstance(signature, Mapping):
        return False
    if str(signature.get("alg", "")).lower() != SIGNATURE_ALGORITHM:
        return False

    covered = signature.get("files")
    manifest = entry.get("files")
    if not isinstance(covered, list) or not all(
        isinstance(item, str) for item in covered
    ):
        return False
    if not isinstance(manifest, list) or not all(
        isinstance(item, str) for item in manifest
    ):
        return False
    if _names(covered) != _names(manifest):
        return False

    value = signature.get("value")
    if not isinstance(value, str) or not value:
        return False
    if key is None or key == "":
        return False

    key_bytes = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    expected = hmac.new(
        key_bytes, _signature_payload(entry), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, value.strip().lower())


def validate_entry(entry: Mapping[str, Any]) -> list[str]:
    """Return problems with the registry entry itself.

    An entry is only loadable when it carries every field needed for the
    integrity chain and its signature covers the complete manifest.
    """
    issues: list[str] = []

    if not isinstance(entry.get("name"), str) or not entry.get("name"):
        issues.append("missing name")
    if not isinstance(entry.get("version"), str) or not entry.get("version"):
        issues.append("missing version")
    if not isinstance(entry.get("digest"), str) or not entry.get("digest"):
        issues.append("missing digest")
    if entry.get("format") not in SUPPORTED_FORMATS:
        issues.append("invalid format")

    files = entry.get("files")
    if not isinstance(files, list) or not files:
        issues.append("missing manifest")

    signature = entry.get("signature")
    if not isinstance(signature, Mapping):
        issues.append("missing signature")
    else:
        covered = signature.get("files")
        if not isinstance(covered, list) or not all(
            isinstance(item, str) for item in covered
        ):
            issues.append("invalid signature file list")
        elif isinstance(files, list) and _names(covered) != _names(files):
            issues.append("signature does not cover manifest")
    return issues


def _validate_model(model: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    """Validate a single model against its own registry entry."""
    registry = policy.get("registry") or {}
    key = policy.get("key")
    name = model.get("name")
    if not isinstance(name, str) or not name:
        return False
    entry = registry.get(name)
    if not isinstance(entry, Mapping):
        # Unknown origin: no registry record means the source cannot be
        # trusted, so the model must never reach the loading stage.
        return False

    if validate_entry(entry):
        return False
    if not verify_signature(entry, key):
        return False
    if not check_digest(model, entry):
        return False
    if not check_source(model, entry, policy):
        return False
    if not check_version(model, entry, policy):
        return False
    if not check_format(model, entry):
        return False
    if check_manifest(model.get("files", []), entry):
        return False
    return True


def model_batch(
    models: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate a batch of models, one registry entry per model.

    Every model is judged independently against its own registry entry.  No
    state is shared between iterations - in particular a digest or verdict
    from one model is never reused for another - and any exception raised
    while judging one model only fails that model, never its neighbours and
    never the batch as a whole.
    """
    results: dict[str, bool] = {}

    for model in models:
        if not isinstance(model, Mapping):
            continue
        name = str(model.get("name", ""))
        try:
            results[name] = _validate_model(model, policy)
        except Exception:
            # Fault isolation: one malformed descriptor must not mask or
            # alter the verdict of any other model in the batch.
            results[name] = False

    return results
