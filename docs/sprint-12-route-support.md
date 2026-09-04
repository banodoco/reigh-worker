# Sprint 12 Route Support

| Route | WGP | VibeComfy | Template |
| --- | --- | --- | --- |
| `z_image_turbo` | supported | replaced_by_astrid_d3 (Worker fail-closed) | `image/z_image` (historical) |
| `z_image_turbo_i2i` | supported | supported | `image/z_image_img2img` |
| `qwen_image_2512` | supported | supported | `image/qwen_image_2512` |
| `qwen_image` | supported | supported | `image/qwen_image_2512` |
| `qwen_image_edit` | supported | supported | `edit/qwen_image_edit` |
| `qwen_image_style` | supported | supported | `edit/qwen_image_edit` |
| `image_inpaint` | supported | supported | `edit/qwen_image_edit` |
| `annotated_image_edit` | supported | supported | `edit/qwen_image_edit` |
| `travel_orchestrator` | supported | wgp-only |  |
| `join_clips_orchestrator` | supported | wgp-only |  |
| `edit_video_orchestrator` | supported | wgp-only |  |
| `travel_segment` | supported | wgp_only (Worker fail-closed) |  |
| `individual_travel_segment` | supported | wgp_only (Worker fail-closed) |  |
| `join_clips_segment` | supported | wgp_only (Worker fail-closed) |  |
| `travel_stitch` | supported | wgp-only |  |
| `join_final_stitch` | supported | wgp-only |  |
| `wan_2_2_t2i` | supported | supported | `video/wanvideo_wrapper_22_14b_t2i` |
| `wan_2_2_i2v` | supported | supported | `video/wanvideo_wrapper_22_14b_i2v_kijai` |
| `image-upscale` | supported | replaced_by_astrid_d4 (Worker fail-closed) | `image/basic_image_upscale` (historical) |
| `image_upscale` | supported | replaced_by_astrid_d4 (Worker fail-closed) | `image/basic_image_upscale` (historical) |
| `video_enhance` | supported | supported | `video/basic_video_enhance` |
| `animate_character` | supported | supported | `video/wan22_animate_native_first_stage` |
| `flux_klein_edit` | supported | supported | `edit/flux2_klein_4b_image_edit_distilled` |
| `travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_i2v_kijai` |
| `travel_segment__model-wan22_vace__guidance-vace_flow__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-vace_flow__continuity-video_source__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-vace_canny__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-vace_canny__continuity-video_source__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-vace_depth__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-vace_depth__continuity-video_source__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-vace_raw__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-vace_raw__continuity-video_source__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `individual_travel_segment__model-wan22_vace__guidance-vace__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `individual_travel_segment__model-wan22_vace__guidance-vace_flow__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `individual_travel_segment__model-wan22_vace__guidance-vace_canny__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `individual_travel_segment__model-wan22_vace__guidance-vace_depth__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `individual_travel_segment__model-wan22_vace__guidance-vace_raw__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default` | supported | vibecomfy_supported | `video/wanvideo_wrapper_22_14b_vace_cocktail` |
| `travel_segment__model-wan22_vace__guidance-uni3c__continuity-first_last__profile-default` | supported | vibecomfy_unsupported |  |
| `travel_segment__model-ltx2__guidance-none__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/ltx2_3_runexx_first_last_frame` |
| `travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/ltx2_3_runexx_first_last_frame` |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_video__continuity-first_last__profile-default` | supported | vibecomfy_unsupported |  |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_pose__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/ltx2_3_first_last_frame_travel_iclora_control` |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_depth__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/ltx2_3_first_last_frame_travel_iclora_control` |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_canny__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/ltx2_3_first_last_frame_travel_iclora_control` |
| `travel_segment__model-ltx2_distilled__guidance-ltx_control_cameraman__continuity-first_last__profile-default` | supported | vibecomfy_supported | `video/ltx2_3_first_last_frame_travel_iclora_control` |
| `animate_character` | `video/wan22_animate_native_first_stage` |
