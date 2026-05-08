# Sprint 12 Route Inventory

| Route | State |
| --- | --- |
| `z_image_turbo` | `dual_supported` |
| `z_image_turbo` | selector |
| `z_image_turbo_i2i` | selector |
| `qwen_image_2512` | selector |
| `qwen_image` | selector |
| `qwen_image_edit` | selector |
| `qwen_image_style` | selector |
| `image_inpaint` | selector |
| `annotated_image_edit` | selector |
| `travel_orchestrator` | selector |
| `join_clips_orchestrator` | selector |
| `edit_video_orchestrator` | selector |
| `travel_segment` | selector |
| `individual_travel_segment` | selector |
| `join_clips_segment` | selector |
| `travel_stitch` | selector |
| `join_final_stitch` | selector |
| `wan_2_2_t2i` | selector |
| `wan_2_2_i2v` | selector |
| `image-upscale` | selector |
| `image_upscale` | selector |
| `video_enhance` | selector |
| `animate_character` | selector |
| `flux_klein_edit` | selector |
| `z_image` | alias |
| `z_image_turbo` | alias |
| `z_image_turbo_i2i` | alias |
| `qwen_image` | alias |
| `qwen_image_2512` | alias |
| `optimised_t2i` | alias |
| `wan_2_2_t2i` | alias |
| `qwen_image_edit` | alias |
| `qwen_image_style` | alias |
| `image_inpaint` | alias |
| `annotated_image_edit` | alias |
| `travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace_flow__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace_flow__continuity-video_source__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace_canny__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace_canny__continuity-video_source__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace_depth__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace_depth__continuity-video_source__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace_raw__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace_raw__continuity-video_source__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default` | NEW |  |
| `individual_travel_segment__model-wan22_vace__guidance-vace__continuity-first_last__profile-default` | NEW |  |
| `individual_travel_segment__model-wan22_vace__guidance-vace_flow__continuity-first_last__profile-default` | NEW |  |
| `individual_travel_segment__model-wan22_vace__guidance-vace_canny__continuity-first_last__profile-default` | NEW |  |
| `individual_travel_segment__model-wan22_vace__guidance-vace_depth__continuity-first_last__profile-default` | NEW |  |
| `individual_travel_segment__model-wan22_vace__guidance-vace_raw__continuity-first_last__profile-default` | NEW |  |
| `join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default` | NEW |  |
| `travel_segment__model-wan22_vace__guidance-uni3c__continuity-first_last__profile-default` | NEW | Requires the NEW Wan 2.2 VACE cocktail template and Uni3C patch before promotion. |
| `travel_segment__model-ltx2__guidance-none__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_video__continuity-first_last__profile-default` | BLOCKED | Raw LTX video guidance uses Wan2GP VG semantics without IC-LoRA; needs a dedicated VibeComfy raw-guide first/last template before promotion. |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_pose__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_depth__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_canny__continuity-first_last__profile-default` | NEW |  |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_cameraman__continuity-first_last__profile-default` | NEW |  |
| z_image_turbo | `z_image_turbo` |
| z_image_alias | `z_image_turbo` |
| wan_2_2_t2i | `wan_2_2_t2i` |
| animate_character | `animate_character` |
| travel_orchestrator | `travel_orchestrator` |
| travel_segment_wan22_vace_raw_video_source | `travel_segment__model-wan2_2_vace__guidance-vace_raw__continuity-video_source__profile-default` |
| travel_segment_ltx_first_last | `travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default` |
