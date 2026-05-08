from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    AppCapability,
    CapabilityContract,
    IMPLEMENTATIONS,
    LiveCase,
    VALID_STATUSES,
    ValidationMessage,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_CONTRACTS_PATH = PACKAGE_DIR / "contracts.json"
DEFAULT_APP_CAPABILITIES_PATH = PACKAGE_DIR / "app_capabilities.json"
DEFAULT_LIVE_MATRIX_MANIFEST_PATH = PACKAGE_DIR / "live_matrix_manifest.json"
WORKER_TEMPLATE_ROUTING_PATH = REPO_ROOT / "source" / "task_handlers" / "tasks" / "template_routing.py"
DEFAULT_APP_ROOT = REPO_ROOT.parent / "reigh-app"
DEFAULT_VIBECOMFY_PATH = REPO_ROOT.parent / "vibecomfy"
DEFAULT_LIVE_REPORTS_DIR = REPO_ROOT / "scripts" / "live_test" / "runs"
VIBECOMFY_TEMPLATE_INDEX_RELATIVE = Path("template_index.json")
VIBECOMFY_COVERAGE_RELATIVE = Path("workflow_corpus") / "manifests" / "coverage.json"


@dataclass(frozen=True)
class WorkerRouteEntry:
    route_key: str
    support_state: str
    template_id: str | None = None
    source_map: str | None = None
    disposition: str | None = None
    blocking_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_key": self.route_key,
            "support_state": self.support_state,
            "template_id": self.template_id,
            "source_map": self.source_map,
            "disposition": self.disposition,
            "blocking_reason": self.blocking_reason,
        }


@dataclass(frozen=True)
class RouteContractIndex:
    by_route_key: Mapping[str, CapabilityContract]
    canonical_by_route_key: Mapping[str, str]
    duplicate_route_keys: Mapping[str, tuple[str, ...]]

    def contract_for_route_key(self, route_key: str) -> CapabilityContract | None:
        return self.by_route_key.get(route_key)


@dataclass(frozen=True)
class AppInventoryEvidence:
    app_root: Path
    capabilities: tuple[AppCapability, ...]
    missing_sources: Mapping[str, str]
    missing_literals: Mapping[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "app_root": str(self.app_root),
            "capability_count": len(self.capabilities),
            "missing_sources": dict(sorted(self.missing_sources.items())),
            "missing_literals": {
                key: list(value)
                for key, value in sorted(self.missing_literals.items())
            },
        }


@dataclass(frozen=True)
class VibeComfyManifestEvidence:
    root: Path
    template_index_path: Path
    coverage_path: Path
    template_ids: frozenset[str]
    workflow_ids: frozenset[str]
    template_index_loaded: bool
    coverage_loaded: bool

    def template_exists(self, template_id: str | None) -> bool:
        return bool(template_id and template_id in self.template_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "template_index_path": str(self.template_index_path),
            "coverage_path": str(self.coverage_path),
            "template_index_loaded": self.template_index_loaded,
            "coverage_loaded": self.coverage_loaded,
            "template_count": len(self.template_ids),
            "workflow_count": len(self.workflow_ids),
        }


@dataclass(frozen=True)
class LiveReportResult:
    case_name: str
    task_type: str | None
    task_id: str | None
    final_status: str | None
    output_location: str | None
    report_path: Path
    generated_at: str | None = None
    error_summary: str | None = None

    @property
    def passed(self) -> bool:
        return self.final_status == "Complete" and bool(self.output_location)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "task_type": self.task_type,
            "task_id": self.task_id,
            "final_status": self.final_status,
            "output_location": self.output_location,
            "report_path": str(self.report_path),
            "generated_at": self.generated_at,
            "error_summary": self.error_summary,
        }


@dataclass(frozen=True)
class LiveReportEvidence:
    reports_dir: Path
    latest_by_case_name: Mapping[str, LiveReportResult]
    report_count: int

    def result_for_case(self, case_name: str) -> LiveReportResult | None:
        return self.latest_by_case_name.get(case_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reports_dir": str(self.reports_dir),
            "report_count": self.report_count,
            "passing_case_count": sum(
                1 for result in self.latest_by_case_name.values() if result.passed
            ),
            "case_count": len(self.latest_by_case_name),
        }


def load_json(path: Path, *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_object(path: Path, *, required: bool = True) -> Mapping[str, Any]:
    data = load_json(path, required=required)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_json_array(path: Path, *, required: bool = True) -> list[Any]:
    data = load_json(path, required=required)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def load_contracts(path: Path = DEFAULT_CONTRACTS_PATH, *, required: bool = False) -> list[CapabilityContract]:
    raw = load_json(path, required=required)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        items = raw.get("contracts", [])
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a contracts array")
    contracts: list[CapabilityContract] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: contracts[{index}] must be an object")
        contracts.append(CapabilityContract.from_mapping(item))
    return contracts


def load_app_capabilities(
    path: Path = DEFAULT_APP_CAPABILITIES_PATH,
    *,
    required: bool = False,
) -> list[AppCapability]:
    raw = load_json(path, required=required)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        items = raw.get("capabilities", [])
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a capabilities array")
    capabilities: list[AppCapability] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: capabilities[{index}] must be an object")
        capabilities.append(AppCapability.from_mapping(item))
    return capabilities


def load_app_root_from_inventory(
    path: Path = DEFAULT_APP_CAPABILITIES_PATH,
    *,
    fallback: Path = DEFAULT_APP_ROOT,
) -> Path:
    raw = load_json(path, required=False)
    if isinstance(raw, Mapping):
        app_root = raw.get("app_root")
        if isinstance(app_root, str) and app_root.strip():
            candidate = Path(app_root)
            if not candidate.is_absolute():
                candidate = (REPO_ROOT / candidate).resolve()
            return candidate
    return fallback


def load_live_cases(
    path: Path = DEFAULT_LIVE_MATRIX_MANIFEST_PATH,
    *,
    required: bool = False,
) -> list[LiveCase]:
    raw = load_json(path, required=required)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        items = raw.get("cases", [])
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a cases array")
    cases: list[LiveCase] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: cases[{index}] must be an object")
        cases.append(LiveCase.from_mapping(item))
    return cases


def scan_app_inventory(
    capabilities: Sequence[AppCapability],
    *,
    app_root: Path | None = None,
) -> AppInventoryEvidence:
    root = app_root or DEFAULT_APP_ROOT
    missing_sources: dict[str, str] = {}
    missing_literals: dict[str, tuple[str, ...]] = {}
    for capability in capabilities:
        source_path = root / capability.source_path
        if not source_path.exists():
            missing_sources[capability.capability_id] = str(source_path)
            continue
        text = source_path.read_text(encoding="utf-8")
        missing = tuple(
            literal
            for literal in capability.expected_literals
            if literal not in text
        )
        if missing:
            missing_literals[capability.capability_id] = missing
    return AppInventoryEvidence(
        app_root=root,
        capabilities=tuple(capabilities),
        missing_sources=missing_sources,
        missing_literals=missing_literals,
    )


def resolve_vibecomfy_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    env_path = os.environ.get("VIBECOMFY_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_VIBECOMFY_PATH


def load_vibecomfy_manifest_evidence(path: Path | None = None) -> VibeComfyManifestEvidence:
    root = resolve_vibecomfy_path(path)
    template_index_path = root / VIBECOMFY_TEMPLATE_INDEX_RELATIVE
    coverage_path = root / VIBECOMFY_COVERAGE_RELATIVE
    template_ids: set[str] = set()
    workflow_ids: set[str] = set()
    template_index_loaded = False
    coverage_loaded = False

    template_index = load_json(template_index_path, required=False)
    if isinstance(template_index, Mapping):
        template_index_loaded = True
        templates = template_index.get("templates", [])
        if isinstance(templates, Mapping):
            template_ids.update(str(key) for key in templates)
        elif isinstance(templates, list):
            for item in templates:
                if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                    template_ids.add(str(item["id"]))

    coverage = load_json(coverage_path, required=False)
    if isinstance(coverage, Mapping):
        coverage_loaded = True
        workflows = coverage.get("workflows", [])
        if isinstance(workflows, Mapping):
            workflow_ids.update(str(key) for key in workflows)
        elif isinstance(workflows, list):
            for item in workflows:
                if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                    workflow_ids.add(str(item["id"]))

    return VibeComfyManifestEvidence(
        root=root,
        template_index_path=template_index_path,
        coverage_path=coverage_path,
        template_ids=frozenset(template_ids),
        workflow_ids=frozenset(workflow_ids),
        template_index_loaded=template_index_loaded,
        coverage_loaded=coverage_loaded,
    )


def load_live_report_evidence(reports_dir: Path = DEFAULT_LIVE_REPORTS_DIR) -> LiveReportEvidence:
    latest_by_case_name: dict[str, LiveReportResult] = {}
    report_count = 0
    for report_path in sorted(reports_dir.glob("*/report.json")):
        raw = load_json(report_path, required=False)
        if not isinstance(raw, Mapping):
            continue
        report_count += 1
        generated_at = _optional_report_str(raw.get("generated_at"))
        results = raw.get("results", [])
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, Mapping):
                continue
            case_name = _optional_report_str(item.get("case_name"))
            if case_name is None:
                continue
            latest_by_case_name[case_name] = LiveReportResult(
                case_name=case_name,
                task_type=_optional_report_str(item.get("task_type")),
                task_id=_optional_report_str(item.get("task_id")),
                final_status=_optional_report_str(item.get("final_status")),
                output_location=_optional_report_str(item.get("output_location")),
                report_path=report_path,
                generated_at=generated_at,
                error_summary=_optional_report_str(item.get("error_summary")),
            )
    return LiveReportEvidence(
        reports_dir=reports_dir,
        latest_by_case_name=latest_by_case_name,
        report_count=report_count,
    )


def build_route_contract_index(contracts: Iterable[CapabilityContract]) -> RouteContractIndex:
    by_route_key: dict[str, CapabilityContract] = {}
    canonical_by_route_key: dict[str, str] = {}
    seen: dict[str, list[str]] = {}
    for contract in contracts:
        for route_key in contract.all_route_keys():
            seen.setdefault(route_key, []).append(contract.capability_id)
            if route_key not in by_route_key:
                by_route_key[route_key] = contract
                canonical_by_route_key[route_key] = contract.canonical_route_key
    duplicates = {
        route_key: tuple(capability_ids)
        for route_key, capability_ids in seen.items()
        if len(capability_ids) > 1
    }
    return RouteContractIndex(
        by_route_key=by_route_key,
        canonical_by_route_key=canonical_by_route_key,
        duplicate_route_keys=duplicates,
    )


def load_worker_route_entries() -> list[WorkerRouteEntry]:
    """Load worker route maps without importing live-test modules."""

    template_routing = _load_template_routing_module()

    entries: list[WorkerRouteEntry] = []
    for source_map, route_map in (
        ("SPRINT_2_SELECTOR_MAP", template_routing.SPRINT_2_SELECTOR_MAP),
        ("SECTION3A_ROUTE_SUPPORT_MAP", template_routing.SECTION3A_ROUTE_SUPPORT_MAP),
    ):
        for route_key, entry in route_map.items():
            entries.append(
                WorkerRouteEntry(
                    route_key=str(route_key),
                    support_state=entry.support_state.value,
                    template_id=entry.template_id,
                    source_map=source_map,
                    disposition=entry.disposition,
                    blocking_reason=entry.blocking_reason,
                )
            )
    return entries


def load_worker_route_aliases() -> dict[str, str]:
    """Return worker direct route aliases without importing live-test modules."""

    template_routing = _load_template_routing_module()

    return {str(alias): str(canonical) for alias, canonical in template_routing.DIRECT_ROUTE_ALIASES.items()}


def _load_template_routing_module() -> ModuleType:
    """Execute template_routing.py directly to avoid broader package imports."""

    module_name = "_capability_contracts_template_routing"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, WORKER_TEMPLATE_ROUTING_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {WORKER_TEMPLATE_ROUTING_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_contract_foundations(contracts: Iterable[CapabilityContract]) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    index = build_route_contract_index(contracts)
    for route_key, capability_ids in sorted(index.duplicate_route_keys.items()):
        messages.append(
            ValidationMessage(
                level="error",
                category="contracts",
                code="duplicate_route_key",
                route_key=route_key,
                message="route key is claimed by multiple capabilities: " + ", ".join(capability_ids),
            )
        )

    for contract in contracts:
        if contract.status not in VALID_STATUSES:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="contracts",
                    code="invalid_status",
                    capability_id=contract.capability_id,
                    message=f"unknown status {contract.status!r}",
                )
            )
        if contract.implementation not in IMPLEMENTATIONS:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="contracts",
                    code="invalid_implementation",
                    capability_id=contract.capability_id,
                    message=f"unknown implementation {contract.implementation!r}",
                )
            )
        if contract.implementation == "vibecomfy" and not contract.template_id:
            messages.append(
                ValidationMessage(
                    level="warning",
                    category="contracts",
                    code="missing_template_id",
                    capability_id=contract.capability_id,
                    route_key=contract.canonical_route_key,
                    message="vibecomfy contracts should name a VibeComfy template id",
                )
            )
    return messages


def summarize_status(contracts: Iterable[CapabilityContract]) -> dict[str, Any]:
    contracts_list = list(contracts)
    by_status: dict[str, int] = {}
    by_implementation: dict[str, int] = {}
    for contract in contracts_list:
        by_status[contract.status] = by_status.get(contract.status, 0) + 1
        by_implementation[contract.implementation] = by_implementation.get(contract.implementation, 0) + 1
    return {
        "contract_count": len(contracts_list),
        "by_status": dict(sorted(by_status.items())),
        "by_implementation": dict(sorted(by_implementation.items())),
    }


def validate_capability_evidence(
    *,
    contracts: Sequence[CapabilityContract],
    app_capabilities: Sequence[AppCapability],
    live_cases: Sequence[LiveCase],
    worker_routes: Sequence[WorkerRouteEntry],
    worker_aliases: Mapping[str, str],
    app_inventory: AppInventoryEvidence,
    vibecomfy: VibeComfyManifestEvidence,
    live_reports: LiveReportEvidence,
) -> list[ValidationMessage]:
    messages = list(validate_contract_foundations(contracts))
    index = build_route_contract_index(contracts)
    app_ids = {capability.capability_id for capability in app_capabilities}
    live_by_case = {case.case_id: case for case in live_cases}
    live_case_ids_by_route: dict[str, set[str]] = {}
    for case in live_cases:
        live_case_ids_by_route.setdefault(case.route_key, set()).add(case.case_id)

    for entry in sorted(worker_routes, key=lambda item: item.route_key):
        contract = index.contract_for_route_key(entry.route_key)
        if contract is None:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="route_binding",
                    code="missing_route_contract",
                    route_key=entry.route_key,
                    message=f"worker route {entry.route_key!r} has no capability contract",
                )
            )
            continue
        if entry.support_state == "vibecomfy_supported":
            if contract.implementation != "vibecomfy":
                messages.append(
                    ValidationMessage(
                        level="error",
                        category="route_binding",
                        code="implementation_mismatch",
                        capability_id=contract.capability_id,
                        route_key=entry.route_key,
                        message="vibecomfy_supported worker route must map to a vibecomfy contract",
                    )
                )
            if contract.template_id != entry.template_id:
                messages.append(
                    ValidationMessage(
                        level="error",
                        category="route_binding",
                        code="template_mismatch",
                        capability_id=contract.capability_id,
                        route_key=entry.route_key,
                        message=(
                            f"contract template {contract.template_id!r} does not match "
                            f"worker template {entry.template_id!r}"
                        ),
                    )
                )
        if entry.support_state == "wgp_only" and contract.implementation != "wgp":
            messages.append(
                ValidationMessage(
                    level="error",
                    category="route_binding",
                    code="implementation_mismatch",
                    capability_id=contract.capability_id,
                    route_key=entry.route_key,
                    message="wgp_only worker route must map to a wgp contract",
                )
            )
        if entry.support_state == "vibecomfy_unsupported" and contract.status != "unsupported_fail_closed":
            messages.append(
                ValidationMessage(
                    level="error",
                    category="route_binding",
                    code="unsupported_not_fail_closed",
                    capability_id=contract.capability_id,
                    route_key=entry.route_key,
                    message="vibecomfy_unsupported worker route must be explicitly fail-closed",
                )
            )

    for alias, canonical in sorted(worker_aliases.items()):
        if alias == canonical:
            continue
        alias_contract = index.contract_for_route_key(alias)
        canonical_contract = index.contract_for_route_key(canonical)
        if canonical_contract is None:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="aliases",
                    code="alias_missing_canonical_contract",
                    route_key=canonical,
                    message=f"route alias {alias!r} points at uncovered canonical route {canonical!r}",
                )
            )
        elif alias_contract is None:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="aliases",
                    code="alias_not_claimed",
                    capability_id=canonical_contract.capability_id,
                    route_key=alias,
                    message=f"route alias {alias!r} is not claimed by the canonical contract",
                )
            )
        elif alias_contract.capability_id != canonical_contract.capability_id:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="aliases",
                    code="alias_contract_mismatch",
                    capability_id=alias_contract.capability_id,
                    route_key=alias,
                    message=(
                        f"route alias {alias!r} maps to {alias_contract.capability_id}, "
                        f"not canonical {canonical_contract.capability_id}"
                    ),
                )
            )

    if not vibecomfy.template_index_loaded:
        messages.append(
            ValidationMessage(
                level="error",
                category="vibecomfy_static_evidence",
                code="missing_template_index",
                path=str(vibecomfy.template_index_path),
                message="VibeComfy template_index.json could not be loaded",
            )
        )
    if not vibecomfy.coverage_loaded:
        messages.append(
            ValidationMessage(
                level="error",
                category="vibecomfy_static_evidence",
                code="missing_coverage_manifest",
                path=str(vibecomfy.coverage_path),
                message="VibeComfy workflow_corpus/manifests/coverage.json could not be loaded",
            )
        )

    for capability_id, path in sorted(app_inventory.missing_sources.items()):
        messages.append(
            ValidationMessage(
                level="error",
                category="app_inventory",
                code="missing_app_source",
                capability_id=capability_id,
                path=path,
                message="configured read-only reigh-app source file is missing",
            )
        )
    for capability_id, literals in sorted(app_inventory.missing_literals.items()):
        messages.append(
            ValidationMessage(
                level="error",
                category="app_inventory",
                code="missing_app_literals",
                capability_id=capability_id,
                message="configured read-only app source is missing expected literals: " + ", ".join(literals),
            )
        )

    for case in sorted(live_cases, key=lambda item: item.case_id):
        contract = index.contract_for_route_key(case.route_key)
        if contract is None:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="live_evidence",
                    code="live_case_route_uncovered",
                    route_key=case.route_key,
                    message=f"live manifest case {case.case_id!r} references an uncovered route",
                )
            )

    for contract in contracts:
        _validate_contract_shape(messages, contract, app_ids=app_ids)
        if contract.implementation == "vibecomfy":
            if not contract.template_id:
                messages.append(
                    ValidationMessage(
                        level="error",
                        category="vibecomfy_static_evidence",
                        code="missing_template_reference",
                        capability_id=contract.capability_id,
                        route_key=contract.canonical_route_key,
                        message="vibecomfy contract must include a template_id",
                    )
                )
            elif not vibecomfy.template_exists(contract.template_id):
                level = "warning" if _contract_records_static_blocker(contract, contract.template_id) else "error"
                messages.append(
                    ValidationMessage(
                        level=level,
                        category="vibecomfy_static_evidence",
                        code="template_index_missing_template",
                        capability_id=contract.capability_id,
                        route_key=contract.canonical_route_key,
                        message=(
                            f"template_id {contract.template_id!r} is absent from "
                            f"{vibecomfy.template_index_path}; VibeComfy owns template validation"
                        ),
                    )
                )
        if contract.status == "runtime_green":
            _validate_runtime_green_evidence(
                messages,
                contract,
                live_by_case=live_by_case,
                live_reports=live_reports,
            )
        if contract.implementation == "vibecomfy" and contract.status in {"runtime_green", "static_valid"}:
            if not contract.static_evidence:
                messages.append(
                    ValidationMessage(
                        level="error",
                        category="vibecomfy_static_evidence",
                        code="missing_static_evidence",
                        capability_id=contract.capability_id,
                        route_key=contract.canonical_route_key,
                        message="vibecomfy runtime/static contracts must name static evidence or blockers",
                    )
                )
        if contract.canonical_route_key in live_case_ids_by_route:
            missing = sorted(live_case_ids_by_route[contract.canonical_route_key] - set(contract.live_evidence))
            if missing and contract.status == "runtime_green":
                messages.append(
                    ValidationMessage(
                        level="error",
                        category="live_evidence",
                        code="manifest_case_not_referenced",
                        capability_id=contract.capability_id,
                        route_key=contract.canonical_route_key,
                        message="runtime_green contract does not reference manifest cases: " + ", ".join(missing),
                    )
                )

    for case_id in sorted(set().union(*(set(contract.live_evidence) for contract in contracts)) - set(live_by_case)):
        messages.append(
            ValidationMessage(
                level="error",
                category="live_evidence",
                code="unknown_live_evidence",
                message=f"contract references unknown live evidence case {case_id!r}",
            )
        )

    return sorted(messages, key=lambda item: (item.level, item.category, item.code, item.capability_id or "", item.route_key or "", item.message))


def _validate_contract_shape(
    messages: list[ValidationMessage],
    contract: CapabilityContract,
    *,
    app_ids: set[str],
) -> None:
    if not contract.variant_axes:
        messages.append(
            ValidationMessage(
                level="error",
                category="variant_accounting",
                code="missing_variant_axes",
                capability_id=contract.capability_id,
                route_key=contract.canonical_route_key,
                message="contract must list variant axes or explicit fail-closed rules",
            )
        )
    artifact = contract.artifact_contract
    if artifact.output_kinds == () and contract.status != "unsupported_fail_closed":
        messages.append(
            ValidationMessage(
                level="error",
                category="artifact_contracts",
                code="missing_output_kinds",
                capability_id=contract.capability_id,
                route_key=contract.canonical_route_key,
                message="non-unsupported contracts must name output artifact kinds",
            )
        )
    if not artifact.db_state:
        messages.append(
            ValidationMessage(
                level="error",
                category="artifact_contracts",
                code="missing_db_state",
                capability_id=contract.capability_id,
                route_key=contract.canonical_route_key,
                message="contract must name DB state/artifact semantics",
            )
        )
    if artifact.storage_paths == () and contract.status != "unsupported_fail_closed":
        messages.append(
            ValidationMessage(
                level="error",
                category="artifact_contracts",
                code="missing_storage_paths",
                capability_id=contract.capability_id,
                route_key=contract.canonical_route_key,
                message="non-unsupported contracts must name storage path semantics",
            )
        )
    if not contract.app_refs:
        messages.append(
            ValidationMessage(
                level="error",
                category="app_inventory",
                code="missing_app_refs",
                capability_id=contract.capability_id,
                route_key=contract.canonical_route_key,
                message="contract must reference app inventory rows or an explicit app exposure exception",
            )
        )
    for app_ref in contract.app_refs:
        if app_ref not in app_ids:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="app_inventory",
                    code="unknown_app_ref",
                    capability_id=contract.capability_id,
                    route_key=contract.canonical_route_key,
                    message=f"contract references unknown app inventory row {app_ref!r}",
                )
            )


def _validate_runtime_green_evidence(
    messages: list[ValidationMessage],
    contract: CapabilityContract,
    *,
    live_by_case: Mapping[str, LiveCase],
    live_reports: LiveReportEvidence,
) -> None:
    if not contract.live_evidence:
        messages.append(
            ValidationMessage(
                level="error",
                category="live_evidence",
                code="runtime_green_missing_live_evidence",
                capability_id=contract.capability_id,
                route_key=contract.canonical_route_key,
                message="runtime_green contract must reference at least one live manifest case",
            )
        )
        return
    for case_id in contract.live_evidence:
        case = live_by_case.get(case_id)
        if case is None:
            continue
        report_case_names = case.report_case_names or (case.case_id,)
        passed = any(
            (live_reports.result_for_case(case_name) is not None and live_reports.result_for_case(case_name).passed)
            for case_name in report_case_names
        )
        if not passed:
            messages.append(
                ValidationMessage(
                    level="error",
                    category="live_evidence",
                    code="runtime_green_without_passing_report",
                    capability_id=contract.capability_id,
                    route_key=contract.canonical_route_key,
                    message=f"live evidence case {case_id!r} has no latest passing report.json result",
                )
            )


def _contract_records_static_blocker(contract: CapabilityContract, template_id: str) -> bool:
    marker = f"vibecomfy_template_index_missing:{template_id}"
    return any(marker in value for value in (*contract.static_evidence, *contract.blockers, *contract.notes))


def _optional_report_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)
