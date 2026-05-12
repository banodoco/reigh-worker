import sys
import types

import numpy as np

from source.media.structure import preprocessors


class _FakePoseAnnotator:
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, frames):
        return [np.zeros_like(frame) for frame in frames]


def _install_fake_pose_module(monkeypatch):
    fake_pose_module = types.ModuleType("Wan2GP.preprocessing.dwpose.pose")
    fake_pose_module.PoseBodyFaceVideoAnnotator = _FakePoseAnnotator
    monkeypatch.setitem(sys.modules, "Wan2GP.preprocessing.dwpose.pose", fake_pose_module)
    monkeypatch.setattr(preprocessors.Path, "exists", lambda self: True)


def test_get_structure_preprocessor_pose_returns_callable(monkeypatch):
    _install_fake_pose_module(monkeypatch)

    pose_preprocessor = preprocessors.get_structure_preprocessor("pose")
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    processed = pose_preprocessor(frames)

    assert callable(pose_preprocessor)
    assert len(processed) == 1
    assert processed[0].shape == frames[0].shape


def test_process_structure_frames_accepts_pose(monkeypatch):
    _install_fake_pose_module(monkeypatch)

    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]

    processed = preprocessors.process_structure_frames(
        frames=frames,
        structure_type="pose",
        motion_strength=1.0,
        canny_intensity=1.0,
        depth_contrast=1.0,
    )

    assert len(processed) == len(frames)


def test_pose_preprocessor_downloads_declared_dwpose_assets(monkeypatch, tmp_path):
    downloaded: list[tuple[str, str, str]] = []

    def _fake_hf_hub_download(*, repo_id, filename, local_dir, subfolder):
        downloaded.append((repo_id, subfolder, filename))
        path = tmp_path / "Wan2GP" / "ckpts" / subfolder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model", encoding="utf-8")
        return str(path)

    _install_fake_pose_module(monkeypatch)
    monkeypatch.setattr(preprocessors, "_wan2gp_dir", lambda: tmp_path / "Wan2GP")
    monkeypatch.setattr(preprocessors.Path, "exists", lambda self: self.is_file() or self.is_dir())
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=_fake_hf_hub_download),
    )

    pose_preprocessor = preprocessors.get_structure_preprocessor("pose")

    assert callable(pose_preprocessor)
    assert downloaded == [
        ("DeepBeepMeep/Wan2.1", "pose", "yolox_l.onnx"),
        ("DeepBeepMeep/Wan2.1", "pose", "dw-ll_ucoco_384.onnx"),
    ]


def test_ensure_dwpose_models_reports_missing_after_failed_stage(monkeypatch, tmp_path):
    def _fake_hf_hub_download(*, repo_id, filename, local_dir, subfolder):
        return str(tmp_path / "Wan2GP" / "ckpts" / subfolder / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=_fake_hf_hub_download),
    )

    try:
        preprocessors.ensure_dwpose_models(tmp_path / "Wan2GP")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("ensure_dwpose_models should fail when staged files are still missing")

    assert "Missing DWPose models required for pose preprocessing" in message
    assert "yolox_l.onnx" in message
    assert "dw-ll_ucoco_384.onnx" in message
