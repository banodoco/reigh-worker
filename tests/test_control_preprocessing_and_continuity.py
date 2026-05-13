from __future__ import annotations

import numpy as np
import pytest

from source.media.structure import preprocessors
from source.task_handlers.tasks.template_routing import (
    WorkerBackend,
    resolve_task_route,
)


def _frames(count: int = 3) -> list[np.ndarray]:
    return [
        np.full((2, 2, 3), idx * 10, dtype=np.uint8)
        for idx in range(count)
    ]


def test_canny_depth_pose_raw_flow_and_uni3c_preprocessors_preserve_frame_count_and_modifiers(monkeypatch):
    frames = _frames(3)
    monkeypatch.setattr(preprocessors.Path, "exists", lambda _self: True)

    class _Canny:
        def __init__(self, _cfg):
            pass

        def forward(self, input_frames):
            return [np.full_like(frame, 100) for frame in input_frames]

    class _Depth:
        def __init__(self, _cfg):
            pass

        def forward(self, input_frames):
            return [np.full_like(frame, 100) for frame in input_frames]

    class _Pose:
        def __init__(self, _cfg):
            pass

        def forward(self, input_frames):
            return [frame + 1 for frame in input_frames]

    class _Flow:
        def __init__(self, _cfg):
            pass

        def forward(self, input_frames):
            return [np.ones_like(input_frames[0]) * 2, np.ones_like(input_frames[0]) * 4], None

    class _FlowViz:
        @staticmethod
        def flow_to_image(flow):
            return flow.astype(np.uint8)

    monkeypatch.setattr(preprocessors, "get_canny_video_annotator_class", lambda: _Canny)
    monkeypatch.setattr(preprocessors, "get_depth_v2_video_annotator_class", lambda: _Depth)
    monkeypatch.setattr(preprocessors, "get_pose_body_face_video_annotator_class", lambda: _Pose)
    monkeypatch.setattr(preprocessors, "get_flow_annotator_class", lambda: _Flow)
    monkeypatch.setattr(preprocessors, "get_flow_viz_module", lambda: _FlowViz)

    canny = preprocessors.process_structure_frames(
        frames, "canny", motion_strength=1.0, canny_intensity=1.5, depth_contrast=1.0
    )
    depth = preprocessors.process_structure_frames(
        frames, "depth", motion_strength=1.0, canny_intensity=1.0, depth_contrast=2.0
    )
    pose = preprocessors.process_structure_frames(
        frames, "pose", motion_strength=1.0, canny_intensity=1.0, depth_contrast=1.0
    )
    flow = preprocessors.process_structure_frames(
        frames, "flow", motion_strength=0.5, canny_intensity=1.0, depth_contrast=1.0
    )
    raw = preprocessors.process_structure_frames(
        frames, "raw", motion_strength=1.0, canny_intensity=1.0, depth_contrast=1.0
    )
    uni3c = preprocessors.process_structure_frames(
        frames, "uni3c", motion_strength=1.0, canny_intensity=1.0, depth_contrast=1.0
    )

    assert len(canny) == len(frames)
    assert int(canny[0][0, 0, 0]) == 150
    assert len(depth) == len(frames)
    assert int(depth[0][0, 0, 0]) == 72
    assert len(pose) == len(frames)
    assert np.array_equal(pose[1], frames[1] + 1)
    assert len(flow) == len(frames)
    assert [int(frame[0, 0, 0]) for frame in flow] == [1, 1, 2]
    assert raw is frames
    assert uni3c is frames


def test_preprocessor_count_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(
        preprocessors,
        "get_structure_preprocessor",
        lambda *_args, **_kwargs: lambda input_frames: input_frames[:1],
    )

    with pytest.raises(ValueError, match="returned 1 frames for 3 input frames"):
        preprocessors.process_structure_frames(
            _frames(3), "canny", motion_strength=1.0, canny_intensity=1.0, depth_contrast=1.0
        )


def test_ltx_control_video_row_uses_raw_vg_template():
    resolved = resolve_task_route(
        task_id="ltx-control-video",
        task_type="travel_segment",
        params={
            "model_name": "ltx2_22B_distilled",
            "continuity_case": "first_last",
            "travel_guidance": {
                "kind": "ltx_control",
                "mode": "video",
                "videos": [{"path": "/tmp/video.mp4"}],
            },
        },
        backend="vibecomfy",
    )

    assert resolved.backend is WorkerBackend.VIBECOMFY
    assert resolved.template_id == "video/ltx2_3_runexx_first_last_raw_video_guide"
    assert resolved.should_use_vibecomfy is True
    assert resolved.fail_closed_reason is None
