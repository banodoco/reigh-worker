from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


STATUS_VOCABULARY: tuple[str, ...] = (
    "unknown",
    "inventoried",
    "routed",
    "template_bound",
    "static_valid",
    "runtime_green",
    "parity_reviewed",
    "app_ready",
    "wgp_only_contract_validated",
    "unsupported_fail_closed",
)
VALID_STATUSES = frozenset(STATUS_VOCABULARY)

IMPLEMENTATIONS = frozenset({"wgp", "vibecomfy", "unsupported"})
MESSAGE_LEVELS = frozenset({"error", "warning", "info"})


@dataclass(frozen=True)
class ValidationMessage:
    level: str
    category: str
    code: str
    message: str
    capability_id: str | None = None
    route_key: str | None = None
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "level": self.level,
            "category": self.category,
            "code": self.code,
            "message": self.message,
        }
        if self.capability_id is not None:
            payload["capability_id"] = self.capability_id
        if self.route_key is not None:
            payload["route_key"] = self.route_key
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class VariantAxis:
    name: str
    values: tuple[str, ...] = ()
    coverage: str | None = None
    fail_closed: bool = False
    notes: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VariantAxis":
        values = _string_tuple(value.get("values"))
        return cls(
            name=_required_str(value, "name"),
            values=values,
            coverage=_optional_str(value.get("coverage")),
            fail_closed=bool(value.get("fail_closed", False)),
            notes=_optional_str(value.get("notes")),
        )

    def as_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "name": self.name,
                "values": list(self.values),
                "coverage": self.coverage,
                "fail_closed": self.fail_closed,
                "notes": self.notes,
            }
        )


@dataclass(frozen=True)
class ArtifactContract:
    output_kinds: tuple[str, ...] = ()
    db_state: tuple[str, ...] = ()
    storage_paths: tuple[str, ...] = ()
    notes: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ArtifactContract":
        source = value or {}
        return cls(
            output_kinds=_string_tuple(source.get("output_kinds")),
            db_state=_string_tuple(source.get("db_state")),
            storage_paths=_string_tuple(source.get("storage_paths")),
            notes=_optional_str(source.get("notes")),
        )

    def as_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "output_kinds": list(self.output_kinds),
                "db_state": list(self.db_state),
                "storage_paths": list(self.storage_paths),
                "notes": self.notes,
            }
        )


@dataclass(frozen=True)
class CapabilityContract:
    capability_id: str
    name: str
    status: str
    implementation: str
    canonical_route_key: str
    route_keys: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    template_id: str | None = None
    variant_axes: tuple[VariantAxis, ...] = ()
    artifact_contract: ArtifactContract = field(default_factory=ArtifactContract)
    app_refs: tuple[str, ...] = ()
    live_evidence: tuple[str, ...] = ()
    static_evidence: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapabilityContract":
        variant_axes = tuple(
            VariantAxis.from_mapping(item)
            for item in _mapping_sequence(value.get("variant_axes"))
        )
        return cls(
            capability_id=_required_str(value, "capability_id"),
            name=_required_str(value, "name"),
            status=_required_str(value, "status"),
            implementation=_required_str(value, "implementation"),
            canonical_route_key=_required_str(value, "canonical_route_key"),
            route_keys=_route_keys(value),
            task_types=_string_tuple(value.get("task_types")),
            template_id=_optional_str(value.get("template_id")),
            variant_axes=variant_axes,
            artifact_contract=ArtifactContract.from_mapping(_optional_mapping(value.get("artifact_contract"))),
            app_refs=_string_tuple(value.get("app_refs")),
            live_evidence=_string_tuple(value.get("live_evidence")),
            static_evidence=_string_tuple(value.get("static_evidence")),
            notes=_string_tuple(value.get("notes")),
            blockers=_string_tuple(value.get("blockers")),
        )

    def all_route_keys(self) -> tuple[str, ...]:
        keys = [self.canonical_route_key, *self.route_keys]
        deduped: list[str] = []
        for key in keys:
            if key and key not in deduped:
                deduped.append(key)
        return tuple(deduped)

    def as_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "capability_id": self.capability_id,
                "name": self.name,
                "status": self.status,
                "implementation": self.implementation,
                "canonical_route_key": self.canonical_route_key,
                "route_keys": list(self.route_keys),
                "task_types": list(self.task_types),
                "template_id": self.template_id,
                "variant_axes": [axis.as_dict() for axis in self.variant_axes],
                "artifact_contract": self.artifact_contract.as_dict(),
                "app_refs": list(self.app_refs),
                "live_evidence": list(self.live_evidence),
                "static_evidence": list(self.static_evidence),
                "notes": list(self.notes),
                "blockers": list(self.blockers),
            }
        )


@dataclass(frozen=True)
class AppCapability:
    capability_id: str
    source_path: str
    expected_literals: tuple[str, ...] = ()
    resolver_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AppCapability":
        return cls(
            capability_id=_required_str(value, "capability_id"),
            source_path=_required_str(value, "source_path"),
            expected_literals=_string_tuple(value.get("expected_literals")),
            resolver_ids=_string_tuple(value.get("resolver_ids")),
            notes=_string_tuple(value.get("notes")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "source_path": self.source_path,
            "expected_literals": list(self.expected_literals),
            "resolver_ids": list(self.resolver_ids),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LiveCase:
    case_id: str
    route_key: str
    task_type: str | None = None
    expected_backend: str | None = None
    report_case_names: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LiveCase":
        return cls(
            case_id=_required_str(value, "case_id"),
            route_key=_required_str(value, "route_key"),
            task_type=_optional_str(value.get("task_type")),
            expected_backend=_optional_str(value.get("expected_backend")),
            report_case_names=_string_tuple(value.get("report_case_names")),
            notes=_string_tuple(value.get("notes")),
        )

    def as_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "case_id": self.case_id,
                "route_key": self.route_key,
                "task_type": self.task_type,
                "expected_backend": self.expected_backend,
                "report_case_names": list(self.report_case_names),
                "notes": list(self.notes),
            }
        )


def _required_str(value: Mapping[str, Any], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(f"missing required string field: {key}")
    return candidate.strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError(f"expected string list, got {type(value).__name__}")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"expected string list item, got {type(item).__name__}")
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return tuple(result)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"expected object list, got {type(value).__name__}")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"expected object list item, got {type(item).__name__}")
        result.append(item)
    return tuple(result)


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"expected object, got {type(value).__name__}")
    return value


def _route_keys(value: Mapping[str, Any]) -> tuple[str, ...]:
    keys = list(_string_tuple(value.get("route_keys")))
    canonical = _required_str(value, "canonical_route_key")
    return tuple(key for key in keys if key != canonical)


def _drop_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
