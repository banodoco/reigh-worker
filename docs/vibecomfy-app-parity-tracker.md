# VibeComfy App Parity Tracker

Last updated: 2026-05-08

This document tracks the app-facing generation paths that are being moved to
VibeComfy, the live validation evidence for each path, and the remaining work
needed before we can claim absolute parity with the existing Wan2GP path.

## Source References

| Area | Source |
| --- | --- |
| Live matrix definitions | `scripts/live_test/matrix.py` |
| Live reports | `scripts/live_test/runs/*/report.md` |
| Worker route selection | `source/task_handlers/tasks/template_routing.py` |
| VibeComfy execution adapter and scratchpads | `source/models/comfy/vibecomfy_adapter.py` |
| WGP direct queue dispatch | `source/task_handlers/tasks/task_registry.py` |
| Wan2GP model/orchestration path | `source/models/wgp/orchestrator.py`, `source/models/model_handlers/qwen_handler.py`, `Wan2GP/` |
| Wan2GP vendor seam | `source/runtime/wgp_bridge.py`, `source/runtime/wgp_ports/vendor_imports.py` |
| VibeComfy template source | `../vibecomfy/ready_templates`, `../vibecomfy/template_index.json` |

## Current Live Validation Matrix

Every `vibecomfy_supported` case in the current live matrix has at least one
successful live report with an output artifact. This is live-green evidence,
not full parity proof across every app parameter permutation.

| Case | Task Type | VibeComfy Template | Latest Passing Report | Status |
| --- | --- | --- | --- | --- |
| `individual_travel_segment_wan22_vace` | `individual_travel_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260507T220856Z/report.md` | live-green |
| `individual_travel_segment_wan22_vace_flow` | `individual_travel_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260507T235320Z/report.md` | live-green |
| `individual_travel_segment_wan22_vace_canny` | `individual_travel_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260507T235320Z/report.md` | live-green |
| `individual_travel_segment_wan22_vace_depth` | `individual_travel_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260507T235320Z/report.md` | live-green |
| `travel_segment_wan22_vace_raw_video_source` | `travel_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260508T003727Z/report.md` | live-green |
| `travel_segment_wan22_vace_flow_video_source` | `travel_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260508T003727Z/report.md` | live-green |
| `travel_segment_wan22_vace_canny_video_source` | `travel_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260508T003727Z/report.md` | live-green |
| `travel_segment_wan22_vace_depth_video_source` | `travel_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260508T003727Z/report.md` | live-green |
| `join_clips_segment_wan22_vace` | `join_clips_segment` | `video/wanvideo_wrapper_22_14b_vace_cocktail` | `scripts/live_test/runs/20260508T013338Z/report.md` | live-green |
| `qwen_image_style` | `qwen_image_style` | `edit/qwen_image_edit` | `scripts/live_test/runs/20260507T223010Z/report.md` | live-green |
| `qwen_image_t2i` | `qwen_image` | `image/qwen_image_2512` | `scripts/live_test/runs/20260508T204310Z/report.md` | live-green |
| `qwen_image_2512` | `qwen_image_2512` | `image/qwen_image_2512` | `scripts/live_test/runs/20260508T205544Z/report.md` | live-green |
| `wan_2_2_t2i` | `wan_2_2_t2i` | `video/wanvideo_wrapper_22_14b_t2i` | `scripts/live_test/runs/20260507T220856Z/report.md` | live-green |
| `wan_2_2_i2v` | `wan_2_2_i2v` | `video/wanvideo_wrapper_22_14b_i2v_kijai` | `scripts/live_test/runs/20260508T113336Z/report.md` | live-green |
| `animate_character` | `animate_character` | `video/wan22_animate_native_first_stage` | `scripts/live_test/runs/20260508T174952Z/report.md` | live-green |
| `image_upscale` | `image-upscale` | `image/basic_image_upscale` | `scripts/live_test/runs/20260508T210920Z/report.md` | live-green |
| `video_enhance` | `video_enhance` | `video/basic_video_enhance` | `scripts/live_test/runs/20260508T202249Z/report.md` | live-green |
| `flux_klein_edit` | `flux_klein_edit` | `edit/flux2_klein_4b_image_edit_distilled` | `scripts/live_test/runs/20260508T085430Z/report.md` | live-green |
| `qwen_image_edit` | `qwen_image_edit` | `edit/qwen_image_edit` | `scripts/live_test/runs/20260508T205544Z/report.md` | live-green |
| `image_inpaint` | `image_inpaint` | `edit/qwen_image_edit` | `scripts/live_test/runs/20260507T223010Z/report.md` | live-green |
| `annotated_image_edit` | `annotated_image_edit` | `edit/qwen_image_edit` | `scripts/live_test/runs/20260507T223010Z/report.md` | live-green |
| `z_image_turbo` | `z_image_turbo` | `image/z_image` | `scripts/live_test/runs/20260508T205544Z/report.md` | live-green |
| `z_image_turbo_i2i` | `z_image_turbo_i2i` | `image/z_image_img2img` | `scripts/live_test/runs/20260507T220856Z/report.md` | live-green |

## Still WGP-Only Or Non-VibeComfy Paths

These are app-relevant paths that are not yet VibeComfy parity claims:

| Case | Task Type | Current State | Notes |
| --- | --- | --- | --- |
| `travel_orchestrator_wan2_1seg` | `travel_orchestrator` | WGP-only | Parent orchestration still creates child tasks and route contracts outside VibeComfy. |
| `travel_orchestrator_ltx` | `travel_orchestrator` | WGP-only | LTX orchestration remains Wan2GP/worker specific. |
| `travel_stitch` | `travel_stitch` | WGP-only | ffmpeg/output stitching path; not a VibeComfy generation template. |
| `join_clips_orchestrator` | `join_clips_orchestrator` | WGP-only | Parent orchestration remains outside VibeComfy. |
| `edit_video_orchestrator` | `edit_video_orchestrator` | WGP-only | Parent orchestration remains outside VibeComfy. |
| `join_final_stitch` | `join_final_stitch` | WGP-only | Final stitch/finalization path. |

Some of these may never need to execute as Comfy workflows, but they still need
parity contracts: they must create the same children, pass the same route
metadata, consume the same artifacts, and produce the same database state.

## Known Parity Gaps

These gaps are intentionally guarded today by fail-closed route logic or by the
current test matrix being representative rather than exhaustive.

| Gap | Current Behavior | Needed For Absolute Parity |
| --- | --- | --- |
| Dynamic user LoRAs | Some VibeComfy routes reject dynamic LoRA params. | General LoRA materialization and injection into VibeComfy templates, including URL download, naming, multiplier schedules, and per-model LoRA directories. |
| Wan I2V LoRA slot count | VibeComfy route rejects more than the supported user slots. | Slot expansion or template/tooling support for the same LoRA capacity Wan2GP accepts. |
| Flux Klein variants | VibeComfy path is pinned to the validated 4B template. | Either validate the other app-selectable variants or keep explicit product-level blocking. |
| Full parameter permutation coverage | Live matrix validates representative settings. | Generate app-derived route cases for resolution, frames, FPS, seed, prompt/negative prompt, control method, source media, LoRA, and profile variations. |
| Parent/child orchestration parity | Child generation paths are green, but parent orchestration is still WGP-only. | Contract tests and live tests proving parent tasks create equivalent child route contracts and final DB/artifact state. |
| Output artifact contract | Output discovery has been patched case-by-case. | A standard output contract per workflow: expected file type, storage prefix, generation row, task output, metadata, and validation probe. |
| VibeComfy template provenance | Templates have been adapted from ready templates and scratchpads. | Each template needs a source note: original workflow/template reference, local adaptation, required custom nodes/models, and parity-relevant settings. |

## Path To Absolute App Parity

1. Build an app route inventory from `reigh-app` and Supabase functions.
   The source of truth should be what the app can actually enqueue, not only
   the worker live matrix.

2. For every app route, generate a parity row:
   task type, app inputs, resolved route key, selected backend, selected
   template, Wan2GP handler/defaults, VibeComfy scratchpad/template, required
   custom nodes, required models, expected output contract, and live evidence.

3. Compare Wan2GP and VibeComfy settings field-by-field:
   model, sampler, scheduler, steps, CFG/guidance, seed, resolution, frame
   count, FPS, prompt, negative prompt, source media, control method, LoRA
   names, LoRA strengths, LoRA schedules, interpolation/postprocess, and output
   encoding.

4. Convert all workflow execution to Python scratchpad/template format.
   Raw JSON workflows should be import sources only. Runtime execution should
   use typed Python mutation so required inputs, outputs, and validation can be
   inspected before queueing.

5. Add a validator that fails before live execution when required Comfy
   connections, required input files, required models, custom nodes, or output
   save nodes are missing.

6. Add a report generator that merges:
   live matrix definitions, route support map, VibeComfy template metadata, and
   latest live reports into this tracker automatically.

7. Promote a route only when it passes all gates:
   route contract test, scratchpad validation, local/static template validation,
   live worker run, artifact contract validation, and Wan2GP parity review.

## Definition Of Done For A Route

A route is absolute-parity ready only when all of these are true:

- It is present in the app route inventory.
- The route key is deterministic and covered by tests.
- Wan2GP source behavior is documented with file/function references.
- VibeComfy template source and local adaptation are documented.
- All app-exposed params either map to VibeComfy behavior or fail closed with a
  product-approved reason.
- Required custom nodes and models are declared and restorable.
- Required workflow connections and save/output nodes are validated statically.
- The live matrix has a passing report with an artifact.
- The artifact contract is verified: task output, generation row, storage
  object, file type, and media probe.
- A regression test would fail if a future change silently routes to a weaker
  or different path.
