from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(REPO_ROOT)
_ADDED_REPO_ROOT = _REPO_ROOT_STR not in sys.path
if _ADDED_REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT_STR)

from scripts.capability_contracts import report
from scripts.capability_contracts.loaders import (
    build_route_contract_index,
    load_app_capabilities,
    load_app_root_from_inventory,
    load_contracts,
    load_live_cases,
    load_worker_route_aliases,
    load_worker_route_entries,
    scan_app_inventory,
)

if _ADDED_REPO_ROOT:
    sys.path.remove(_REPO_ROOT_STR)


TRACKER_PATH = REPO_ROOT / "docs" / "vibecomfy-app-parity-tracker.md"
PARITY_CRITICAL_VARIANT_AXES = {
    "lora",
    "control",
    "source_media",
    "resolution",
    "frames_fps",
    "profile",
    "postprocess",
}
TOOL_README_PATH = REPO_ROOT / "scripts" / "capability_contracts" / "README.md"


def test_worker_route_maps_are_covered_by_contracts_or_aliases() -> None:
    contracts = load_contracts()
    index = build_route_contract_index(contracts)
    missing = [
        route.route_key
        for route in load_worker_route_entries()
        if index.contract_for_route_key(route.route_key) is None
    ]

    assert missing == []


def test_route_aliases_are_covered_by_one_canonical_capability() -> None:
    index = build_route_contract_index(load_contracts())
    mismatches: list[tuple[str, str, str | None, str | None]] = []

    for alias, canonical in sorted(load_worker_route_aliases().items()):
        alias_contract = index.contract_for_route_key(alias)
        canonical_contract = index.contract_for_route_key(canonical)
        if canonical_contract is None or alias_contract is None:
            mismatches.append(
                (
                    alias,
                    canonical,
                    alias_contract.capability_id if alias_contract else None,
                    canonical_contract.capability_id if canonical_contract else None,
                )
            )
        elif alias_contract.capability_id != canonical_contract.capability_id:
            mismatches.append(
                (
                    alias,
                    canonical,
                    alias_contract.capability_id,
                    canonical_contract.capability_id,
                )
            )

    assert mismatches == []


def test_live_matrix_manifest_matches_build_matrix_stable_fields() -> None:
    from scripts.live_test.matrix import build_matrix

    manifest_by_case = {case.case_id: case for case in load_live_cases()}
    matrix_cases = [
        case
        for case in build_matrix()
        if case.route_key is not None and case.support_state is not None
    ]

    assert sorted(manifest_by_case) == sorted(case.name for case in matrix_cases)
    for case in matrix_cases:
        manifest = manifest_by_case[case.name]
        assert manifest.route_key == case.route_key
        assert manifest.task_type == case.task_type
        assert manifest.expected_backend == _expected_backend(case.support_state)
        assert set(manifest.report_case_names or (manifest.case_id,))


def test_vibecomfy_supported_manifest_cases_have_runtime_evidence_links() -> None:
    contracts = load_contracts()
    index = build_route_contract_index(contracts)
    manifest_cases = load_live_cases()

    missing: list[tuple[str, str, str]] = []
    for case in manifest_cases:
        if case.expected_backend != "vibecomfy":
            continue
        contract = index.contract_for_route_key(case.route_key)
        if contract is None:
            missing.append((case.case_id, case.route_key, "missing_contract"))
            continue
        if contract.implementation != "vibecomfy":
            missing.append((case.case_id, case.route_key, contract.implementation))
        if _is_runtime_green_or_higher(contract.status) and case.case_id not in contract.live_evidence:
            missing.append((case.case_id, case.route_key, "missing_live_evidence"))

    assert missing == []


def test_vibecomfy_contracts_have_template_artifact_variant_app_and_static_evidence() -> None:
    missing: list[tuple[str, str]] = []
    for contract in load_contracts():
        if contract.implementation != "vibecomfy":
            continue
        if not contract.template_id:
            missing.append((contract.capability_id, "template_id"))
        if not contract.variant_axes:
            missing.append((contract.capability_id, "variant_axes"))
        if not contract.app_refs:
            missing.append((contract.capability_id, "app_refs"))
        if not (contract.static_evidence or contract.blockers):
            missing.append((contract.capability_id, "static_evidence_or_blockers"))
        artifact = contract.artifact_contract
        if not artifact.output_kinds:
            missing.append((contract.capability_id, "artifact.output_kinds"))
        if not artifact.db_state:
            missing.append((contract.capability_id, "artifact.db_state"))
        if not artifact.storage_paths:
            missing.append((contract.capability_id, "artifact.storage_paths"))

    assert missing == []


def test_vibecomfy_contracts_do_not_keep_stale_template_index_blockers() -> None:
    stale: list[tuple[str, str]] = []
    for contract in load_contracts():
        if contract.implementation != "vibecomfy" or not contract.template_id:
            continue
        marker = f"vibecomfy_template_index_missing:{contract.template_id}"
        haystack = (*contract.static_evidence, *contract.blockers, *contract.notes)
        if any(marker in value for value in haystack):
            stale.append((contract.capability_id, contract.template_id))
        if any("was not present in ../vibecomfy/template_index.json during seed scan" in value for value in haystack):
            stale.append((contract.capability_id, contract.template_id))

    assert stale == []


def test_contracts_encode_parity_critical_variant_axes() -> None:
    missing: list[tuple[str, str]] = []
    weak: list[tuple[str, str]] = []
    for contract in load_contracts():
        axes = {axis.name: axis for axis in contract.variant_axes}
        for axis_name in sorted(PARITY_CRITICAL_VARIANT_AXES):
            if axis_name not in axes:
                missing.append((contract.capability_id, axis_name))
                continue
            axis = axes[axis_name]
            if not axis.values:
                weak.append((contract.capability_id, f"{axis_name}.values"))
            if not axis.coverage:
                weak.append((contract.capability_id, f"{axis_name}.coverage"))

    assert missing == []
    assert weak == []


def test_wgp_only_contracts_omit_templates_and_include_db_artifact_behavior() -> None:
    missing: list[tuple[str, str]] = []
    for contract in load_contracts():
        if contract.implementation != "wgp":
            continue
        if contract.template_id is not None:
            missing.append((contract.capability_id, "template_id_must_be_empty"))
        if contract.status != "wgp_only_contract_validated":
            missing.append((contract.capability_id, "status"))
        artifact = contract.artifact_contract
        if not artifact.output_kinds:
            missing.append((contract.capability_id, "artifact.output_kinds"))
        if not artifact.db_state:
            missing.append((contract.capability_id, "artifact.db_state"))
        if not artifact.storage_paths:
            missing.append((contract.capability_id, "artifact.storage_paths"))
        if not any("WGP-only" in note for note in contract.notes):
            missing.append((contract.capability_id, "wgp_only_rationale"))
        if not {"child_creation", "route_metadata", "artifact_fan_in"}.issubset(
            {axis.name for axis in contract.variant_axes}
        ):
            missing.append((contract.capability_id, "wgp_behavior_axes"))

    assert missing == []


def test_travel_orchestrator_declares_pose_preprocessor_assets() -> None:
    contracts = {contract.capability_id: contract for contract in load_contracts()}
    travel = contracts["cap.travel_orchestrator"]

    assert {
        "worker_preprocessor_model_asset:Wan2GP/ckpts/pose/yolox_l.onnx",
        "worker_preprocessor_model_asset:Wan2GP/ckpts/pose/dw-ll_ucoco_384.onnx",
    }.issubset(set(travel.static_evidence))


def test_unsupported_contracts_include_fail_closed_rationale() -> None:
    missing: list[tuple[str, str]] = []
    for contract in load_contracts():
        if contract.implementation != "unsupported":
            continue
        if contract.status != "unsupported_fail_closed":
            missing.append((contract.capability_id, "status"))
        if not contract.blockers and not contract.notes:
            missing.append((contract.capability_id, "rationale"))
        rationale = " ".join((*contract.notes, *contract.blockers)).lower()
        if not any(token in rationale for token in ("fail-closed", "fail closed", "unsupported", "no vibecomfy")):
            missing.append((contract.capability_id, "fail_closed_rationale"))
        if not any(axis.fail_closed for axis in contract.variant_axes):
            missing.append((contract.capability_id, "fail_closed_axis"))

    assert missing == []


def test_app_inventory_sources_and_literals_exist_read_only() -> None:
    capabilities = load_app_capabilities()
    app_root = load_app_root_from_inventory()
    scan = scan_app_inventory(capabilities, app_root=app_root)

    assert scan.missing_sources == {}
    assert scan.missing_literals == {}


def test_tracker_render_output_is_deterministic() -> None:
    args = report.build_parser().parse_args(["render"])
    rendered = report.render_markdown(report.load_report_evidence(args))

    assert rendered == report.render_markdown(report.load_report_evidence(args))
    assert TRACKER_PATH.exists()


def test_discoverability_docs_cover_workflow_decision_points_and_commands() -> None:
    readme = TOOL_README_PATH.read_text(encoding="utf-8")
    tracker = TRACKER_PATH.read_text(encoding="utf-8")
    combined = f"{readme}\n{tracker}"

    expected_phrases = (
        "Import existing workflow",
        "Fork workflow",
        "Scratch-built workflow",
        "Validation",
        "App parity",
        "Community-added workflows",
        "route aliases or variant axes",
        "VibeComfy graph/schema/model validation",
    )
    expected_commands = (
        "python -m scripts.capability_contracts.report validate",
        "python -m scripts.capability_contracts.report next-actions",
        "python -m scripts.capability_contracts.report app-inventory --json",
        "python -m scripts.capability_contracts.report render docs/vibecomfy-app-parity-tracker.md",
    )

    missing = [phrase for phrase in (*expected_phrases, *expected_commands) if phrase not in combined]

    assert missing == []


def _expected_backend(support_state: str) -> str:
    if support_state == "vibecomfy_supported":
        return "vibecomfy"
    if support_state == "wgp_only":
        return "wgp"
    if support_state == "vibecomfy_unsupported":
        return "unsupported"
    raise AssertionError(f"unexpected support state: {support_state}")


def _is_runtime_green_or_higher(status: str) -> bool:
    return status in {"runtime_green", "parity_reviewed", "app_ready"}
