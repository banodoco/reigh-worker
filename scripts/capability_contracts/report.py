from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .loaders import (
    DEFAULT_APP_CAPABILITIES_PATH,
    DEFAULT_APP_ROOT,
    DEFAULT_CONTRACTS_PATH,
    DEFAULT_LIVE_MATRIX_MANIFEST_PATH,
    DEFAULT_LIVE_REPORTS_DIR,
    DEFAULT_VIBECOMFY_PATH,
    build_route_contract_index,
    load_app_root_from_inventory,
    load_app_capabilities,
    load_contracts,
    load_live_cases,
    load_live_report_evidence,
    load_vibecomfy_manifest_evidence,
    load_worker_route_aliases,
    load_worker_route_entries,
    scan_app_inventory,
    summarize_status,
    validate_capability_evidence,
)
from .models import STATUS_VOCABULARY, ValidationMessage


ACTION_CATEGORIES: tuple[str, ...] = (
    "app_inventory",
    "route_binding",
    "vibecomfy_static_evidence",
    "live_evidence",
    "artifact_contracts",
    "variant_accounting",
    "aliases",
    "parity_review",
)

CATEGORY_FOR_MESSAGE: dict[str, str] = {
    "app_inventory": "app_inventory",
    "route_binding": "route_binding",
    "vibecomfy_static_evidence": "vibecomfy_static_evidence",
    "live_evidence": "live_evidence",
    "artifact_contracts": "artifact_contracts",
    "variant_accounting": "variant_accounting",
    "aliases": "aliases",
    "contracts": "parity_review",
    "files": "parity_review",
}

COMMAND_HINTS: dict[str, str] = {
    "app_inventory": "Inspect ../reigh-app read-only, then update scripts/capability_contracts/app_capabilities.json or the affected contract app_refs.",
    "route_binding": "Update scripts/capability_contracts/contracts.json to match source/task_handlers/tasks/template_routing.py.",
    "vibecomfy_static_evidence": "In ../vibecomfy, use VibeComfy-owned checks such as `python -m vibecomfy.cli validate` or `python -m vibecomfy.cli doctor`; do not reimplement workflow graph validation in reigh-worker.",
    "live_evidence": "Run or inspect scripts/live_test reports, then update scripts/capability_contracts/live_matrix_manifest.json and contract live_evidence pointers.",
    "artifact_contracts": "Update the contract artifact_contract in scripts/capability_contracts/contracts.json with DB/storage/output semantics.",
    "variant_accounting": "Update contract variant_axes in scripts/capability_contracts/contracts.json with covered values or explicit fail-closed rules.",
    "aliases": "Model aliases on the canonical contract route_keys instead of creating duplicate product capabilities.",
    "parity_review": "Review status/blockers in scripts/capability_contracts/contracts.json and promote status only after static/live evidence is present.",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.capability_contracts.report",
        description="Report worker-local product capability contract parity status.",
    )
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS_PATH)
    parser.add_argument("--app-capabilities", type=Path, default=DEFAULT_APP_CAPABILITIES_PATH)
    parser.add_argument("--live-manifest", type=Path, default=DEFAULT_LIVE_MATRIX_MANIFEST_PATH)
    parser.add_argument("--app-root", type=Path, default=None, help=f"Read-only reigh-app root (default: inventory app_root or {DEFAULT_APP_ROOT})")
    parser.add_argument("--vibecomfy-path", type=Path, default=DEFAULT_VIBECOMFY_PATH, help="VibeComfy checkout containing template_index.json and workflow_corpus/manifests/coverage.json")
    parser.add_argument("--live-runs-dir", type=Path, default=DEFAULT_LIVE_REPORTS_DIR, help="Directory containing scripts/live_test/runs/*/report.json")

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate contract source files and route indexes.")
    validate.add_argument("--json", action="store_true", help="Emit machine-readable validation results.")
    validate.add_argument("--warnings-as-errors", action="store_true", help="Treat warnings as a nonzero validation result.")
    validate.set_defaults(func=cmd_validate)

    status = subparsers.add_parser("status", help="Print deterministic capability status summary.")
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    status.set_defaults(func=cmd_status)

    render = subparsers.add_parser("render", help="Render a markdown parity tracker from loaded data.")
    render.add_argument("output", nargs="?", type=Path)
    render.add_argument("--check", action="store_true", help="Fail if the rendered output differs from the target file.")
    render.set_defaults(func=cmd_render)

    next_actions = subparsers.add_parser("next-actions", help="Print grouped next actions for parity closure.")
    next_actions.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    next_actions.set_defaults(func=cmd_next_actions)

    app_inventory = subparsers.add_parser("app-inventory", help="Print configured read-only app capability inventory.")
    app_inventory.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    app_inventory.set_defaults(func=cmd_app_inventory)

    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    evidence = load_report_evidence(args)
    messages = evidence["messages"]
    payload = validation_payload(messages)
    if args.json:
        _print_json(payload)
    else:
        for message in messages:
            print(_format_message(message), file=sys.stderr if message.level == "error" else sys.stdout)
        if not messages:
            print("No capability-contract validation findings.")
        else:
            print(
                "Validation summary: "
                f"{payload['error_count']} errors, {payload['warning_count']} warnings."
            )
    if args.warnings_as_errors and payload["warning_count"]:
        return 1
    return 1 if payload["error_count"] else 0


def cmd_status(args: argparse.Namespace) -> int:
    evidence = load_report_evidence(args)
    contracts = evidence["contracts"]
    app_capabilities = evidence["app_capabilities"]
    live_cases = evidence["live_cases"]
    worker_routes = evidence["worker_routes"]
    aliases = evidence["worker_aliases"]
    messages = evidence["messages"]
    vibecomfy = evidence["vibecomfy"]
    app_inventory = evidence["app_inventory"]
    live_reports = evidence["live_reports"]
    payload = {
        "status_vocabulary": list(STATUS_VOCABULARY),
        "contracts": summarize_status(contracts),
        "app_inventory_count": len(app_capabilities),
        "live_case_count": len(live_cases),
        "worker_route_count": len(worker_routes),
        "worker_alias_count": len(aliases),
        "app_inventory": app_inventory.as_dict(),
        "vibecomfy_static_evidence": vibecomfy.as_dict(),
        "live_reports": live_reports.as_dict(),
        "validation": validation_payload(messages),
        "next_action_counts": {
            category: len(items)
            for category, items in build_next_actions(evidence).items()
        },
    }
    if args.json:
        _print_json(payload)
    else:
        print(render_status_text(payload))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    evidence = load_report_evidence(args)
    markdown = render_markdown(evidence)
    if args.check:
        if args.output is None:
            print("render --check requires an output path", file=sys.stderr)
            return 2
        expected = args.output.read_text(encoding="utf-8") if args.output.exists() else None
        if expected != markdown:
            print(f"{args.output} is not current", file=sys.stderr)
            return 1
        return 0
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.write_text(markdown, encoding="utf-8")
    return 0


def cmd_next_actions(args: argparse.Namespace) -> int:
    evidence = load_report_evidence(args)
    actions = build_next_actions(evidence)
    if args.json:
        _print_json(actions)
    else:
        for category in ACTION_CATEGORIES:
            items = actions[category]
            print(f"{category}:")
            if items:
                for item in items:
                    print(f"- [{item['level']}] {item['target']}: {item['message']}")
                    print(f"  hint: {item['command_hint']}")
            else:
                print("- no actions")
    return 0


def cmd_app_inventory(args: argparse.Namespace) -> int:
    app_capabilities = load_app_capabilities(args.app_capabilities, required=False)
    app_root = args.app_root or load_app_root_from_inventory(args.app_capabilities)
    scan = scan_app_inventory(app_capabilities, app_root=app_root)
    payload = {
        "capabilities": [capability.as_dict() for capability in app_capabilities],
        "scan": scan.as_dict(),
        "next_actions": _app_inventory_actions(scan.missing_sources, scan.missing_literals),
    }
    if args.json:
        _print_json(payload)
    else:
        if not app_capabilities:
            print("No app capability inventory rows loaded.")
        for capability in app_capabilities:
            print(f"{capability.capability_id}: {capability.source_path}")
        if scan.missing_sources or scan.missing_literals:
            print("App inventory scan findings:")
            for item in _app_inventory_actions(scan.missing_sources, scan.missing_literals):
                print(f"- {item['target']}: {item['message']}")
    return 0


def load_report_evidence(args: argparse.Namespace) -> dict[str, Any]:
    contracts = load_contracts(args.contracts, required=False)
    app_capabilities = load_app_capabilities(args.app_capabilities, required=False)
    live_cases = load_live_cases(args.live_manifest, required=False)
    worker_routes = load_worker_route_entries()
    worker_aliases = load_worker_route_aliases()
    app_root = args.app_root or load_app_root_from_inventory(args.app_capabilities)
    app_inventory = scan_app_inventory(app_capabilities, app_root=app_root)
    vibecomfy = load_vibecomfy_manifest_evidence(args.vibecomfy_path)
    live_reports = load_live_report_evidence(args.live_runs_dir)
    messages = _foundation_messages(args) + validate_capability_evidence(
        contracts=contracts,
        app_capabilities=app_capabilities,
        live_cases=live_cases,
        worker_routes=worker_routes,
        worker_aliases=worker_aliases,
        app_inventory=app_inventory,
        vibecomfy=vibecomfy,
        live_reports=live_reports,
    )
    return {
        "contracts": contracts,
        "app_capabilities": app_capabilities,
        "live_cases": live_cases,
        "worker_routes": worker_routes,
        "worker_aliases": worker_aliases,
        "app_inventory": app_inventory,
        "vibecomfy": vibecomfy,
        "live_reports": live_reports,
        "messages": messages,
    }


def render_status_text(payload: dict[str, Any]) -> str:
    lines = [
        "Capability Contract Status",
        f"contracts: {payload['contracts']['contract_count']}",
        f"app inventory rows: {payload['app_inventory_count']}",
        f"live cases: {payload['live_case_count']}",
        f"worker routes: {payload['worker_route_count']}",
        f"worker aliases: {payload['worker_alias_count']}",
        f"validation errors: {payload['validation']['error_count']}",
        f"validation warnings: {payload['validation']['warning_count']}",
    ]
    action_counts = payload.get("next_action_counts", {})
    if isinstance(action_counts, dict):
        lines.append("next actions:")
        for category in ACTION_CATEGORIES:
            lines.append(f"  {category}: {action_counts.get(category, 0)}")
    return "\n".join(lines)


def render_markdown(evidence: dict[str, Any]) -> str:
    contracts = sorted(evidence["contracts"], key=lambda item: item.capability_id)
    app_capabilities = sorted(evidence["app_capabilities"], key=lambda item: item.capability_id)
    live_cases = sorted(evidence["live_cases"], key=lambda item: item.case_id)
    worker_routes = sorted(evidence["worker_routes"], key=lambda item: item.route_key)
    worker_aliases = dict(sorted(evidence["worker_aliases"].items()))
    app_inventory = evidence["app_inventory"]
    vibecomfy = evidence["vibecomfy"]
    live_reports = evidence["live_reports"]
    validation = validation_payload(evidence["messages"])
    next_actions = build_next_actions(evidence)
    status = summarize_status(contracts)
    live_by_route: dict[str, list[str]] = {}
    for case in live_cases:
        live_by_route.setdefault(case.route_key, []).append(case.case_id)
    app_ref_counts: dict[str, int] = {capability.capability_id: 0 for capability in app_capabilities}
    for contract in contracts:
        for app_ref in contract.app_refs:
            if app_ref in app_ref_counts:
                app_ref_counts[app_ref] += 1
    route_index = build_route_contract_index(contracts)

    lines = [
        "<!-- Generated by python -m scripts.capability_contracts.report render. Do not edit by hand. -->",
        "",
        "# VibeComfy App Parity Tracker",
        "",
        "This tracker is generated from the worker-local capability contract registry and evidence manifests.",
        "Edit `scripts/capability_contracts/contracts.json`, `app_capabilities.json`, or `live_matrix_manifest.json`, then regenerate it.",
        "",
        "## When To Use This Tracker",
        "",
        "- Import, fork, scratch-build, or accept a community workflow when it changes an app-facing capability, route, template binding, variants, artifacts, or parity claim.",
        "- Keep VibeComfy graph/schema/custom-node/model validation in `../vibecomfy`; keep product routes, app inventory, artifact semantics, live evidence, aliases, unsupported variants, and parity status here.",
        "- Prefer adding route aliases or variant axes to an existing capability when product behavior is unchanged. Add a new capability only for new app-facing behavior.",
        "- Run `python -m scripts.capability_contracts.report validate`, then `python -m scripts.capability_contracts.report next-actions`, then `python -m scripts.capability_contracts.report render docs/vibecomfy-app-parity-tracker.md` after edits.",
        "",
        "## Summary",
        "",
        f"- Contracts: {len(contracts)}",
        f"- App inventory rows: {len(app_capabilities)}",
        f"- Live matrix manifest cases: {len(live_cases)}",
        f"- Worker route rows: {len(worker_routes)}",
        f"- Worker route aliases: {len(worker_aliases)}",
        f"- Latest live report files: {live_reports.report_count}",
        f"- Latest passing live report cases: {live_reports.as_dict()['passing_case_count']}",
        f"- Validation errors: {validation['error_count']}",
        f"- Validation warnings: {validation['warning_count']}",
        f"- VibeComfy template index loaded: {_yes_no(vibecomfy.template_index_loaded)} (`{vibecomfy.template_index_path}`)",
        f"- VibeComfy coverage manifest loaded: {_yes_no(vibecomfy.coverage_loaded)} (`{vibecomfy.coverage_path}`)",
        "",
        "### By Status",
        "",
        _markdown_table(
            ("Status", "Count"),
            [(f"`{key}`", str(value)) for key, value in sorted(status["by_status"].items())],
        ),
        "",
        "### By Implementation",
        "",
        _markdown_table(
            ("Implementation", "Count"),
            [(f"`{key}`", str(value)) for key, value in sorted(status["by_implementation"].items())],
        ),
        "",
        "## Status Vocabulary",
        "",
    ]
    lines.extend(_status_legend())
    lines.extend(
        [
            "",
            "## App Coverage",
            "",
            f"Configured app root: `{app_inventory.app_root}`",
            "",
            _markdown_table(
                ("App capability", "Source", "Contracts", "Scan"),
                [
                    (
                        f"`{capability.capability_id}`",
                        f"`{capability.source_path}`",
                        str(app_ref_counts.get(capability.capability_id, 0)),
                        _app_scan_status(capability.capability_id, app_inventory),
                    )
                    for capability in app_capabilities
                ],
            ),
            "",
            "## Capability Contracts",
            "",
            _markdown_table(
                ("Capability", "Status", "Implementation", "Canonical route", "Template", "Live evidence", "Blockers"),
                [
                    (
                        f"`{contract.capability_id}`",
                        f"`{contract.status}`",
                        f"`{contract.implementation}`",
                        f"`{contract.canonical_route_key}`",
                        f"`{contract.template_id}`" if contract.template_id else "",
                        _join_code(contract.live_evidence),
                        _join_text(contract.blockers),
                    )
                    for contract in contracts
                ],
            ),
            "",
            "## Live Evidence",
            "",
            _markdown_table(
                ("Case", "Route", "Task type", "Report names"),
                [
                    (
                        f"`{case.case_id}`",
                        f"`{case.route_key}`",
                        f"`{case.task_type}`" if case.task_type else "",
                        _join_code(case.report_case_names or (case.case_id,)),
                    )
                    for case in live_cases
                ],
            ),
            "",
            "## WGP-Only Paths",
            "",
            _markdown_table(
                ("Capability", "Route", "Artifact contract"),
                [
                    (
                        f"`{contract.capability_id}`",
                        f"`{contract.canonical_route_key}`",
                        _join_text(contract.artifact_contract.output_kinds),
                    )
                    for contract in contracts
                    if contract.implementation == "wgp"
                ],
            ),
            "",
            "## Unsupported Fail-Closed Paths",
            "",
            _markdown_table(
                ("Capability", "Route", "Notes"),
                [
                    (
                        f"`{contract.capability_id}`",
                        f"`{contract.canonical_route_key}`",
                        _join_text((*contract.notes, *contract.blockers)),
                    )
                    for contract in contracts
                    if contract.status == "unsupported_fail_closed"
                ],
            ),
            "",
            "## Alias Coverage",
            "",
            _markdown_table(
                ("Alias", "Canonical", "Contract"),
                [
                    (
                        f"`{alias}`",
                        f"`{canonical}`",
                        f"`{route_index.contract_for_route_key(alias).capability_id}`"
                        if route_index.contract_for_route_key(alias) is not None
                        else "",
                    )
                    for alias, canonical in worker_aliases.items()
                ],
            ),
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = _blocker_rows(contracts, validation)
    lines.append(
        _markdown_table(
            ("Level", "Category", "Target", "Message"),
            blockers,
        )
        if blockers
        else "No blockers recorded."
    )
    lines.extend(
        [
            "",
            "## Variant Accounting",
            "",
            _markdown_table(
                ("Capability", "Axis", "Values", "Coverage", "Fail closed"),
                _variant_rows(contracts),
            ),
            "",
            "## Next Actions",
            "",
        ]
    )
    for category in ACTION_CATEGORIES:
        lines.append(f"### {category}")
        lines.append("")
        actions = next_actions[category]
        if actions:
            lines.extend(
                f"- `{item['level']}` `{item['code']}` `{item['target']}`: {item['message']} Hint: {item['command_hint']}"
                for item in actions
            )
        else:
            lines.append("- No actions.")
        lines.append("")
    return "\n".join(lines)


def _status_legend() -> list[str]:
    descriptions = {
        "unknown": "not yet inventoried or validated",
        "inventoried": "known product capability exists",
        "routed": "worker route decision is accounted for",
        "template_bound": "VibeComfy template binding exists",
        "static_valid": "static/template evidence is present",
        "runtime_green": "latest live evidence has passed",
        "parity_reviewed": "Wan2GP/app parity review has been completed",
        "app_ready": "ready for app-facing parity claims",
        "wgp_only_contract_validated": "non-template WGP/orchestration contract is validated",
        "unsupported_fail_closed": "unsupported variant is intentionally blocked",
    }
    return [f"- `{status}`: {descriptions[status]}" for status in STATUS_VOCABULARY]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_escape_table_cell(str(cell)) for cell in row) + " |"
        for row in rows
    ]
    if not body:
        body = ["| " + " | ".join("" for _ in headers) + " |"]
    return "\n".join([header_line, separator, *body])


def _escape_table_cell(value: str) -> str:
    return value.replace("\n", "<br>").replace("|", "\\|")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _app_scan_status(capability_id: str, app_inventory: Any) -> str:
    if capability_id in app_inventory.missing_sources:
        return "missing source"
    if capability_id in app_inventory.missing_literals:
        return "missing literals: " + ", ".join(app_inventory.missing_literals[capability_id])
    return "ok"


def _join_code(values: Sequence[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def _join_text(values: Sequence[str]) -> str:
    return "<br>".join(values)


def _blocker_rows(
    contracts: Sequence[Any],
    validation: dict[str, Any],
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for contract in contracts:
        for blocker in contract.blockers:
            rows.append(("blocker", "contract", contract.capability_id, blocker))
    for message in validation["messages"]:
        if message["level"] in {"error", "warning"}:
            target = message.get("capability_id") or message.get("route_key") or message.get("path") or "(global)"
            rows.append((message["level"], message["category"], target, message["message"]))
    return sorted(set(rows), key=lambda row: row)


def _variant_rows(contracts: Sequence[Any]) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for contract in contracts:
        for axis in contract.variant_axes:
            rows.append(
                (
                    f"`{contract.capability_id}`",
                    f"`{axis.name}`",
                    _join_code(axis.values),
                    axis.coverage or "",
                    _yes_no(axis.fail_closed),
                )
            )
    return rows


def _foundation_messages(args: argparse.Namespace) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    for path, code in (
        (args.contracts, "missing_contracts_file"),
        (args.app_capabilities, "missing_app_capabilities_file"),
        (args.live_manifest, "missing_live_manifest_file"),
    ):
        if not path.exists():
            messages.append(
                ValidationMessage(
                    level="warning",
                    category="files",
                    code=code,
                    path=str(path),
                    message="source file is not present yet",
                )
            )
    return messages


def build_next_actions(evidence: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    actions: dict[str, list[dict[str, str]]] = {category: [] for category in ACTION_CATEGORIES}
    for message in evidence["messages"]:
        category = CATEGORY_FOR_MESSAGE.get(message.category, "parity_review")
        actions[category].append(_message_to_action(message, category))

    contracts = evidence["contracts"]
    index = build_route_contract_index(contracts)
    for entry in sorted(evidence["worker_routes"], key=lambda item: item.route_key):
        if entry.support_state == "vibecomfy_supported" and index.contract_for_route_key(entry.route_key) is None:
            actions["route_binding"].append(
                {
                    "level": "error",
                    "code": "missing_route_contract",
                    "target": entry.route_key,
                    "message": f"add contract row for supported route {entry.route_key}",
                    "command_hint": COMMAND_HINTS["route_binding"],
                }
            )

    return {
        category: sorted(items, key=lambda item: (item["level"], item["code"], item["target"], item["message"]))
        for category, items in actions.items()
    }


def _message_to_action(message: ValidationMessage, category: str) -> dict[str, str]:
    target = message.capability_id or message.route_key or message.path or "(global)"
    return {
        "level": message.level,
        "code": message.code,
        "target": target,
        "message": message.message,
        "command_hint": COMMAND_HINTS.get(category, COMMAND_HINTS["parity_review"]),
    }


def _app_inventory_actions(
    missing_sources: Any,
    missing_literals: Any,
) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for capability_id, path in sorted(dict(missing_sources).items()):
        actions.append(
            {
                "level": "error",
                "code": "missing_app_source",
                "target": capability_id,
                "message": f"configured source file is missing: {path}",
                "command_hint": COMMAND_HINTS["app_inventory"],
            }
        )
    for capability_id, literals in sorted(dict(missing_literals).items()):
        actions.append(
            {
                "level": "error",
                "code": "missing_app_literals",
                "target": capability_id,
                "message": "configured source file is missing expected literals: " + ", ".join(literals),
                "command_hint": COMMAND_HINTS["app_inventory"],
            }
        )
    return actions


def validation_payload(messages: Sequence[ValidationMessage]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for message in messages:
        bucket = by_category.setdefault(message.category, {"errors": 0, "warnings": 0, "infos": 0})
        if message.level == "error":
            bucket["errors"] += 1
        elif message.level == "warning":
            bucket["warnings"] += 1
        else:
            bucket["infos"] += 1
    return {
        "error_count": sum(1 for message in messages if message.level == "error"),
        "warning_count": sum(1 for message in messages if message.level == "warning"),
        "info_count": sum(1 for message in messages if message.level == "info"),
        "by_category": {key: by_category[key] for key in sorted(by_category)},
        "messages": [message.as_dict() for message in messages],
    }


def _message_summary(messages: Sequence[ValidationMessage]) -> dict[str, Any]:
    return validation_payload(messages)


def _format_message(message: ValidationMessage) -> str:
    parts = [message.level.upper(), message.category, message.code]
    if message.capability_id:
        parts.append(message.capability_id)
    if message.route_key:
        parts.append(message.route_key)
    if message.path:
        parts.append(message.path)
    return ": ".join(parts) + f": {message.message}"


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
