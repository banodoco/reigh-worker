---
name: live-test
description: Launch a Reigh live-test run on RunPod. Picks the prebuilt validation environment when available, falls back to the cold fresh path otherwise. Use whenever the user wants to validate a worker change against vibecomfy parity, drive the matrix harness, or smoke-test a workflow on a real GPU.
---

# Live-test harness — recommended invocation

The Reigh live-test harness has three variants:

- `--variant fresh` — provisions a clean pod, runs ~67 min of cold install (uv
  sync + VibeComfy + custom nodes), then drives the matrix. The historical
  default; still the argparse default for backward compatibility.
- `--variant prebuilt` — provisions a pod with a pre-baked RunPod network
  volume attached, extracts the venv + VibeComfy bundles to `/opt/`, syncs the
  worker/vibecomfy refs, then launches. Reaches `launch_worker` materially
  faster than the cold path.
- `--variant auto` — opt-in: preflights for a prebuilt volume and dispatches
  prebuilt on a hit or fresh on a miss. Emits a structured
  `{"event": "prebuilt_available"|"prebuilt_unavailable", ...}` log line so the
  fallback reason is visible without spelunking through fresh-variant logs.

## Default invocation

**Pass `--variant auto`** for new live-test work. The harness will pick the
prebuilt path when it can and silently fall back to fresh when the volume
isn't there.

```sh
python -m scripts.live_test.main \
  --variant auto \
  --backend vibecomfy \
  --case z_image_turbo
```

The argparse default is still `fresh` so existing automation keeps working;
agents should pass `--variant auto` explicitly.

## When to use `--variant prebuilt` directly

When you've just built a prebuilt cache or know one exists in the region and
want a hard failure (rather than a fresh fallback) if it's missing or drifted:

```sh
python -m scripts.live_test.main \
  --variant prebuilt \
  --backend vibecomfy \
  --case z_image_turbo
```

Flags consumed by the prebuilt variant:

- `--prebuilt-volume-name NAME` — override the auto-discovered volume.
- `--strict-prebuilt` — abort on any drift instead of delta-syncing.
- `--no-allow-delta` — disable delta sync; combine with `--strict-prebuilt`
  to enforce zero drift.
- `--update-manifest-on-sync` — rewrite the manifest after a successful delta
  sync (default: don't rewrite).
- `--container-disk-gb N` — container disk size in GB (floor 100, default 200).
- `--python-version X.Y` — override the expected manifest python_version.

## Building the prebuilt cache

```sh
runpod-lifecycle prebuilt build \
  --volume-name reigh-livetest-prebuilt-portable-eu-no-1 \
  --data-center EU-NO-1 \
  --attention-profile portable \
  --worker-ref main \
  --vibecomfy-ref main \
  --python-version 3.10
```

Other `runpod-lifecycle prebuilt` subcommands: `inspect`, `invalidate`,
`list`. `invalidate` preserves `models/` and `build.lock`.

## Drift rules at a glance

- HARD-FAIL (rebuild required): `schema_version`, `bundle_format_version`,
  `python_version`, `cuda_extra`.
- Delta-sync (reconciled on the consumer): `pyproject_hash`,
  `custom_nodes_lock_hash`, `comfyui_pin`, `vibecomfy_commit`,
  `reigh_worker_commit`.

When the cache is too drifted to reuse:

```sh
runpod-lifecycle prebuilt invalidate --volume-name <name> --data-center <dc>
runpod-lifecycle prebuilt build --volume-name <name> --data-center <dc> --attention-profile portable
```

## variant_update coexistence

`--variant update` aborts with `Prebuilt cache present at /workspace/…` when
a prebuilt manifest is on the attached volume; this applies to both
`--pod-id` and `--spawn-takeover` modes. If you really want to mutate the
prebuilt cache (rare), `rl prebuilt invalidate` first.

## More detail

See [`docs/migration-vibecomfy-live-validation.md`](../../../../docs/migration-vibecomfy-live-validation.md)
for the full architecture, invalidation table, model-cache layout, and the
v1 first-run cold-download caveat.
