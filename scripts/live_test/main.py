"""CLI entrypoint for the live worker harness."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.live_test import config
from scripts.live_test.inspect import main as run_inspect
from scripts.live_test.matrix import build_matrix, build_target_manifest
from scripts.live_test.variant_fresh import run as run_variant_fresh
from scripts.live_test.variant_prebuilt import run as run_variant_prebuilt
from scripts.live_test.variant_update import run as run_variant_update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Reigh live worker harness.")
    # Default stays `fresh` so existing automation continues to work unchanged.
    # `auto` is opt-in: it preflights a prebuilt volume and falls back to fresh.
    parser.add_argument(
        "--variant",
        choices=("fresh", "update", "prebuilt", "auto"),
        default="fresh",
    )
    parser.add_argument("--pod-id", help="Existing RunPod pod ID for update-mode takeover.")
    parser.add_argument(
        "--spawn-takeover",
        action="store_true",
        help="Spawn a fresh orchestrator-managed pod, then take it over with the local worker branch.",
    )
    termination_group = parser.add_mutually_exclusive_group()
    termination_group.add_argument("--no-terminate", dest="no_terminate", action="store_true")
    termination_group.add_argument("--terminate", dest="no_terminate", action="store_false")
    parser.set_defaults(no_terminate=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ref", default="main", help="Branch/ref to clone for Variant Fresh.")
    parser.add_argument(
        "--vibecomfy-ref",
        default="megaplan/production-parity-templates",
        help="VibeComfy branch/ref to clone for VibeComfy backend live tests.",
    )
    parser.add_argument("--wgp-profile", type=int, default=3)
    parser.add_argument(
        "--backend",
        choices=("wgp", "vibecomfy"),
        default="wgp",
        help="Worker backend to inject through REIGH_BACKEND.",
    )
    parser.add_argument(
        "--selector-namespace",
        default="production",
        help="Route selector namespace to inject into worker claim validation.",
    )
    parser.add_argument(
        "--selector-version",
        help="Optional route selector version to inject into worker claim validation.",
    )
    parser.add_argument(
        "--worker-contract-version",
        type=int,
        default=1,
        help="Worker route contract version to stamp into Reigh-shaped live-test tasks.",
    )
    parser.add_argument(
        "--worker-profile",
        default="default",
        help="Selected worker route profile to stamp into route-specific live-test tasks.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Restrict the matrix to a case name. May be passed multiple times.",
    )
    parser.add_argument(
        "--task-type",
        action="append",
        default=[],
        help="Restrict the matrix to a task type. May be passed multiple times.",
    )
    parser.add_argument(
        "--route-key",
        action="append",
        default=[],
        help="Restrict the matrix to a route key. May be passed multiple times.",
    )
    parser.add_argument(
        "--wgp-rollback",
        action="store_true",
        help="Run the selected route/task cases in rollback mode by forcing REIGH_BACKEND=wgp.",
    )
    parser.add_argument(
        "--allow-fresh-heartbeat",
        action="store_true",
        help="Allow update-mode takeover of a pod that is already running this live-test worker.",
    )
    parser.add_argument("--timeout-image", type=int, default=config.TIMEOUT_IMAGE_SEC)
    parser.add_argument(
        "--timeout-travel-segment",
        type=int,
        default=config.TIMEOUT_INDIVIDUAL_TRAVEL_SEGMENT_SEC,
    )
    parser.add_argument(
        "--timeout-travel-orchestrator",
        type=int,
        default=config.TIMEOUT_TRAVEL_ORCHESTRATOR_SEC,
    )
    parser.add_argument("--anchor-image-a", default=config.ANCHOR_IMAGE_A_URL)
    parser.add_argument("--anchor-image-b", default=config.ANCHOR_IMAGE_B_URL)
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Reserved for future matrix fan-out; current harness always runs serially.",
    )
    # --- Prebuilt validation-environment flags (consumed by variant_prebuilt) ----
    parser.add_argument(
        "--prebuilt-volume-name",
        default=None,
        help="Override the prebuilt volume name (defaults to PREBUILT_VOLUME_NAME_PREFIX + profile-).",
    )
    parser.add_argument(
        "--prebuilt-data-center",
        default=None,
        help="Prebuilt variant: only use a prebuilt volume from this RunPod data center.",
    )
    parser.add_argument(
        "--strict-prebuilt",
        action="store_true",
        help="Prebuilt variant: abort on any delta drift instead of delta-syncing.",
    )
    parser.add_argument(
        "--no-allow-delta",
        dest="allow_delta",
        action="store_false",
        help="Disable delta sync; combine with --strict-prebuilt to enforce zero drift.",
    )
    parser.set_defaults(allow_delta=True)
    parser.add_argument(
        "--update-manifest-on-sync",
        action="store_true",
        help="Rewrite the manifest after a successful delta sync (default: do not rewrite).",
    )
    parser.add_argument(
        "--container-disk-gb",
        type=int,
        default=None,
        help="Container disk size in GB for the consumer pod (prebuilt only; floor 100, default 200).",
    )
    parser.add_argument(
        "--python-version",
        default=None,
        help="Override expected python_version for the prebuilt manifest hard-fail drift check.",
    )
    parser.add_argument(
        "--emit-targets-json",
        metavar="PATH",
        help=(
            "Resolve the selected live-test cases into a Reigh target manifest "
            "and exit without DB, RunPod, pod launch, or VibeComfy imports."
        ),
    )
    return parser


def _live_test_selector_namespace() -> str:
    return datetime.now(timezone.utc).strftime("livet-%Y%m%d%H%M%S")


def _finalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Selector-namespace normalisation runs for all variants (fresh / update /
    # prebuilt) so vibecomfy+production gets the live-test-scoped namespace
    # regardless of which driver dispatches.
    if args.wgp_rollback:
        args.backend = "wgp"
    elif args.backend == "vibecomfy" and args.selector_namespace == "production":
        args.selector_namespace = _live_test_selector_namespace()

    if args.variant in ("fresh", "prebuilt", "auto"):
        if args.pod_id or args.spawn_takeover:
            parser.error(
                "--pod-id/--spawn-takeover are only valid with --variant update"
            )
        if args.no_terminate is None:
            args.no_terminate = False
        if args.variant in ("prebuilt", "auto"):
            # Enforce container-disk floor of 100 GB; default to 200 when unset.
            if args.container_disk_gb is None:
                args.container_disk_gb = 200
            if args.container_disk_gb < 100:
                parser.error(
                    f"--container-disk-gb must be >= 100 for --variant {args.variant} "
                    f"(got {args.container_disk_gb})"
                )
        return args

    # update variant
    if bool(args.pod_id) == bool(args.spawn_takeover):
        parser.error("update variant requires exactly one of --pod-id or --spawn-takeover")
    if args.no_terminate is None:
        args.no_terminate = True
    return args


def _auto_dispatch_variant(args: argparse.Namespace) -> str:
    """Preflight a prebuilt volume and return either 'prebuilt' or 'fresh'.

    Emits a structured `prebuilt_unavailable` log line on stdout when no volume
    matches, so the operator can see the fallback reason without spelunking
    through the fresh-variant logs.
    """
    api_key = config.get_env("RUNPOD_API_KEY")
    if not api_key:
        print(
            json.dumps(
                {
                    "event": "prebuilt_unavailable",
                    "reason": "RUNPOD_API_KEY not set; cannot enumerate volumes",
                }
            ),
            flush=True,
        )
        return "fresh"
    try:
        from scripts.live_test._shared import select_network_volume

        selection = select_network_volume(
            api_key,
            name_prefix=config.PREBUILT_VOLUME_NAME_PREFIX,
            data_center_filter=getattr(args, "prebuilt_data_center", None),
        )
    except Exception as exc:  # noqa: BLE001 — auto preflight should never explode
        print(
            json.dumps(
                {
                    "event": "prebuilt_unavailable",
                    "reason": f"select_network_volume failed: {exc}",
                }
            ),
            flush=True,
        )
        return "fresh"
    if selection is None:
        print(
            json.dumps(
                {
                    "event": "prebuilt_unavailable",
                    "reason": (
                        f"no volume matches prefix {config.PREBUILT_VOLUME_NAME_PREFIX!r}"
                    ),
                }
            ),
            flush=True,
        )
        return "fresh"
    _, name, data_center_id = selection
    print(
        json.dumps(
            {
                "event": "prebuilt_available",
                "volume_name": name,
                "data_center_id": data_center_id,
            }
        ),
        flush=True,
    )
    return "prebuilt"


def _emit_targets_json(args: argparse.Namespace) -> int:
    cases = build_matrix(
        anchor_image_a=args.anchor_image_a,
        anchor_image_b=args.anchor_image_b,
        timeout_image_sec=args.timeout_image,
        timeout_travel_segment_sec=args.timeout_travel_segment,
        timeout_travel_orchestrator_sec=args.timeout_travel_orchestrator,
        selected_backend=getattr(args, "backend", "wgp"),
        selector_namespace=getattr(args, "selector_namespace", "production"),
        selector_version=getattr(args, "selector_version", None),
        worker_contract_version=getattr(args, "worker_contract_version", 1),
        selected_profile=getattr(args, "worker_profile", "default"),
        case_names=getattr(args, "case", []),
        task_types=getattr(args, "task_type", []),
        route_keys=getattr(args, "route_key", []),
    )
    manifest = build_target_manifest(
        cases,
        selected_backend=getattr(args, "backend", "wgp"),
        selector_namespace=getattr(args, "selector_namespace", "production"),
        selector_version=getattr(args, "selector_version", None),
        worker_contract_version=getattr(args, "worker_contract_version", 1),
        selected_profile=getattr(args, "worker_profile", "default"),
        selection={
            "case_names": list(getattr(args, "case", [])),
            "task_types": list(getattr(args, "task_type", [])),
            "route_keys": list(getattr(args, "route_key", [])),
        },
    )
    output_path = Path(args.emit_targets_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(output_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv and effective_argv[0] == "inspect":
        return run_inspect(effective_argv[1:])
    parser = build_parser()
    args = _finalize_args(parser.parse_args(effective_argv), parser)
    if args.emit_targets_json:
        return _emit_targets_json(args)
    if args.variant == "auto":
        resolved = _auto_dispatch_variant(args)
        args.variant = resolved
    if args.variant == "fresh":
        return run_variant_fresh(args)
    if args.variant == "prebuilt":
        return run_variant_prebuilt(args)
    return run_variant_update(args)


if __name__ == "__main__":
    raise SystemExit(main())
