"""Specialized task handlers for complex multi-step operations.

Subpackages:
- travel/: Multi-segment travel video generation with SVI chaining
- join/: Video clip joining with AI-generated transitions
- queue/: Task queue and worker thread management
- tasks/: Task registry, type definitions, and conversion
- worker/: Worker utilities, heartbeat, and fatal error handling

Top-level handlers:
- edit_video_orchestrator: Video editing workflow coordination
- magic_edit: AI-powered image editing via Replicate API
- frame-range video inpainting is not a supported Worker entrypoint
- create_visualization: Debug visualization generation
- extract_frame / rife_interpolate: Specialized single-purpose handlers
"""
