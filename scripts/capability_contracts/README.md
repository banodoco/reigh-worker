# Capability Contracts

This package is the worker-local source of truth for app parity accounting. It
tracks product capabilities, route decisions, implementation bindings, artifact
contracts, variant coverage, app source inventory, and live evidence.

VibeComfy owns workflow/template validation. This package consumes VibeComfy
manifests and points developers to VibeComfy commands; it does not reimplement
graph, schema, custom-node, or model checks in `reigh-worker`.

## Files

- `contracts.json`: capability contracts keyed by product behavior.
- `app_capabilities.json`: read-only app source inventory and literals that
  should exist in `../reigh-app`.
- `live_matrix_manifest.json`: stable metadata for live matrix cases without
  importing `scripts.live_test.matrix`.
- `report.py`: deterministic CLI for status, validation, rendering, and next
  actions.

## Commands

```bash
python -m scripts.capability_contracts.report status --json
python -m scripts.capability_contracts.report app-inventory --json
python -m scripts.capability_contracts.report next-actions
python -m scripts.capability_contracts.report validate
python -m scripts.capability_contracts.report render docs/vibecomfy-app-parity-tracker.md
python -m scripts.capability_contracts.report render --check docs/vibecomfy-app-parity-tracker.md
```

Use `VIBECOMFY_PATH=/path/to/vibecomfy` when the VibeComfy checkout is not at
`../vibecomfy`.

## When To Use This

Use capability contracts whenever a workflow changes what the Reigh app can
enqueue or what the worker can route, validate, or claim as app parity.

- Import existing workflow: add or update the product capability in
  `contracts.json`, bind it to a VibeComfy `template_id`, add app refs and
  artifact semantics, then run `python -m scripts.capability_contracts.report validate`.
- Fork workflow: keep the same capability row when the product behavior and
  route stay the same; add route aliases or variant axes instead of duplicating
  a capability. Create a new capability only for new app-facing behavior.
- Scratch-built workflow: draft the contract first with `status` no higher than
  `inventoried` or `routed`, record missing evidence as blockers, and promote
  only after VibeComfy static checks and worker live evidence exist.
- Validation: run `python -m scripts.capability_contracts.report validate` for
  source consistency, then `python -m scripts.capability_contracts.report next-actions`
  to see which repo owns the remaining work.
- App parity: run `python -m scripts.capability_contracts.report app-inventory --json`
  before claiming app coverage, and regenerate
  `docs/vibecomfy-app-parity-tracker.md` with `render` after edits.
- Community-added workflows: treat community workflow IDs as VibeComfy-owned
  inputs. Record the product route, template binding, variants, artifacts, and
  blockers here; run VibeComfy graph/schema/model validation in `../vibecomfy`.

## Add Or Change A Capability

1. Add or update an app inventory row in `app_capabilities.json` with the
   relevant read-only `../reigh-app` source file and expected literals.
2. Add or update a row in `contracts.json` with:
   - stable `capability_id` and product-facing `name`
   - `canonical_route_key` and any `route_keys` aliases
   - implementation: `vibecomfy`, `wgp`, or `unsupported`
   - VibeComfy `template_id` only when implementation is `vibecomfy`
   - explicit `variant_axes`, including covered values and fail-closed rules
   - `artifact_contract` for DB state, storage paths, and output kinds
   - `app_refs`, `live_evidence`, `static_evidence`, notes, and blockers
3. Add or update `live_matrix_manifest.json` only for stable live evidence
   metadata. Do not import live-test modules from report-time code.
4. Run `python -m scripts.capability_contracts.report validate`.
5. Run `python -m scripts.capability_contracts.report next-actions` and close
   worker-local omissions before treating remaining items as app or VibeComfy
   blockers.
6. Regenerate the tracker with `render`.

## Validation Boundary

`reigh-worker` owns product/app parity accounting, route coverage, Wan2GP
behavior references, DB/storage artifact semantics, and live evidence links.

VibeComfy owns workflow/template internals. For template graph/schema/custom
node/model checks, use VibeComfy-owned commands from the VibeComfy checkout:

```bash
cd ../vibecomfy
python -m vibecomfy.cli validate
python -m vibecomfy.cli doctor
python -m vibecomfy.cli workflows list
python -m vibecomfy.cli workflows inspect <workflow-id>
```

If a worker report points at missing VibeComfy template-index evidence, record a
blocker in `contracts.json` or update VibeComfy artifacts in that repo. Do not
patch around VibeComfy validation inside `reigh-worker`.

## App Inventory Expectations

The app inventory is intentionally lightweight. It should prove that a
capability row still points at real app enqueue/configuration truth without
requiring this repo to import, build, or mutate `../reigh-app`.

Each inventory row should name a source file and literals or resolver ids that
would disappear if app enqueue behavior drifted. Missing files or literals are
validation errors.

## Workflow Workbench Direction

Future import, fork, and scratch-built workflow tooling should build on these
contracts:

- draft a capability contract before a workflow is ported
- link imported workflow ids and VibeComfy template ids
- preserve route aliases rather than duplicating product capabilities
- show next actions by app inventory, route binding, static evidence, live
  evidence, artifacts, variants, aliases, and parity review
- keep reusable workflow mechanics in VibeComfy and product/app parity evidence
  in `reigh-worker`
