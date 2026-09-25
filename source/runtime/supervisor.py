"""Launch and supervise the one Astrid GenericPackHost owned by a Worker.

The Worker is deliberately not a task executor. It only validates the
operator-supplied host profile, starts one external GenericPackHost, publishes
neutral process state, forwards termination signals, and returns the host's
exit status unchanged.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


GENERIC_HOST_MODULE = "astrid.core.execution.generic_host"
GENERIC_HOST_EXECUTOR_ID = "astrid-pack-host"
PREPARATION_VERSION = "runtime.local-worker-preparation/v2"
ACTIVATION_VERSION = "runtime.local-worker-activation/v1"
RECEIPT_VERSION = "runtime.local-worker-receipt/v2"
CONTROL_VERSION = "reigh.local-worker-control/v1"
ACTIVATION_ACCEPTED_VERSION = "astrid.local-worker-activation-accepted/v1"
_CONTROL_FRAME_LIMIT = 64 * 1024

# Ambient process settings only. Host bindings that affect identity are
# validated and passed as argv values rather than inherited from the process.
HOST_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
        "TMPDIR",
        "TEMP",
        "TMP",
        "XDG_RUNTIME_DIR",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ASTRID_HOST_READINESS_PROFILE_PATH",
        "ASTRID_HOST_READINESS_PROFILE_HASH",
        # This is a selector input only.  It is forwarded so the host cannot
        # silently lose the requested route, but it is never placement proof.
        "ASTRID_EXECUTION_TARGET_JSON",
    }
)


class LauncherConfigurationError(ValueError):
    """A trusted host binding is missing or unsafe."""


@dataclass(frozen=True)
class RuntimeDiscovery:
    """The immutable, secret-free Runtime record consumed by the Worker."""

    endpoint: str
    port: int
    pid: int
    process_birth_id: str
    runtime_instance_id: str
    coordinator_epoch: str
    active_realm: str
    realm_root: Path
    protocol_version: str
    schema_version: str
    worker_credential_file: Path
    worker_actor: str
    worker_scopes: tuple[str, ...]
    snapshot_digest: str


def _required_env(name: str, environ: Mapping[str, str]) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise LauncherConfigurationError(f"{name} is required")
    return value


def _reject_unissued_execution_target(environ: Mapping[str, str]) -> None:
    """Fail closed when a targeted route has no trusted placement producer.

    ``ASTRID_EXECUTION_TARGET_JSON`` is an operator/request selector.  The
    selected Plan-A route must also receive Runtime-authenticated actual
    placement, account/pod/profile, and incarnation evidence.  The current
    Worker has no producer for that evidence, so allowing the selector to
    reach GenericPackHost would turn configuration into an authority claim.
    """

    raw = environ.get("ASTRID_EXECUTION_TARGET_JSON", "").strip()
    if not raw:
        return
    try:
        selector = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise LauncherConfigurationError(
            "ASTRID_EXECUTION_TARGET_JSON is not valid JSON; trusted placement issuer is unavailable"
        ) from exc
    if not isinstance(selector, Mapping):
        raise LauncherConfigurationError(
            "ASTRID_EXECUTION_TARGET_JSON must be an object; trusted placement issuer is unavailable"
        )
    raise LauncherConfigurationError(
        "targeted Plan-A execution is unavailable: Worker has no credential-backed "
        "placement issuer for actual account/pod/profile/incarnation evidence"
    )


def _resolved_path(
    name: str,
    environ: Mapping[str, str],
    *,
    directory: bool = False,
    executable: bool = False,
    must_exist: bool = True,
) -> Path:
    raw = _required_env(name, environ)
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise LauncherConfigurationError(f"{name} must be an absolute path")
    if not must_exist:
        try:
            parent = candidate.parent.resolve(strict=True)
        except OSError as exc:
            raise LauncherConfigurationError(f"{name} parent directory is unavailable") from exc
        return parent / candidate.name
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LauncherConfigurationError(f"{name} is unavailable") from exc
    if directory:
        if not resolved.is_dir():
            raise LauncherConfigurationError(f"{name} must name a directory")
    elif not resolved.is_file() or (executable and not os.access(resolved, os.X_OK)):
        raise LauncherConfigurationError(f"{name} must name a file")
    return resolved


def _absolute_value(name: str, environ: Mapping[str, str]) -> str:
    value = _required_env(name, environ)
    path = Path(value)
    if not path.is_absolute():
        raise LauncherConfigurationError(f"{name} must be an absolute path")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise LauncherConfigurationError(f"{name} parent directory is unavailable") from exc
    if not parent.is_dir():
        raise LauncherConfigurationError(f"{name} parent must be a directory")
    return str(parent / path.name)


@dataclass(frozen=True)
class HostLaunchConfig:
    """The complete trusted binding for one GenericPackHost."""

    host_python: Path
    source_checkout: Path
    pack_root: Path
    runtime_endpoint: str
    credential_file: Path
    support_root: Path
    runtime_instance_id: str
    ready_file: Path
    state_file: Path
    boot_manifest_path: Path
    boot_manifest_hash: str
    capability_matrix: Path | None = None
    readiness_profile_path: Path | None = None
    readiness_profile_hash: str | None = None

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        parked_credential: bool = False,
    ) -> "HostLaunchConfig":
        env = os.environ if environ is None else environ
        source_checkout = _resolved_path("ASTRID_HOST_SOURCE_CHECKOUT", env, directory=True)
        pack_root = _resolved_path("ASTRID_HOST_PACK_ROOT", env, directory=True)
        if not pack_root.is_relative_to(source_checkout):
            raise LauncherConfigurationError("ASTRID_HOST_PACK_ROOT must be inside ASTRID_HOST_SOURCE_CHECKOUT")
        support_root = _resolved_path("ASTRID_HOST_SUPPORT_ROOT", env, directory=True)
        credential_file = _resolved_path(
            "ASTRID_HOST_CREDENTIAL_FILE", env, must_exist=not parked_credential
        )
        boot_manifest_path = _resolved_path("ASTRID_HOST_BOOT_MANIFEST_PATH", env)
        capability_matrix = None
        if env.get("ASTRID_HOST_CAPABILITY_MATRIX", "").strip():
            capability_matrix = _resolved_path("ASTRID_HOST_CAPABILITY_MATRIX", env)
        return cls(
            host_python=_resolved_path("ASTRID_HOST_PYTHON", env, executable=True),
            source_checkout=source_checkout,
            pack_root=pack_root,
            runtime_endpoint=_required_env("ASTRID_HOST_RUNTIME_ENDPOINT", env).rstrip("/"),
            credential_file=credential_file,
            support_root=support_root,
            runtime_instance_id=_required_env("ASTRID_HOST_RUNTIME_INSTANCE_ID", env),
            ready_file=Path(_absolute_value("ASTRID_HOST_READY_FILE", env)),
            state_file=Path(_absolute_value("ASTRID_HOST_STATE_FILE", env)),
            boot_manifest_path=boot_manifest_path,
            boot_manifest_hash=_required_env("ASTRID_HOST_BOOT_MANIFEST_HASH", env),
            capability_matrix=capability_matrix,
        )

    def argv(
        self,
        *,
        activation_fd: int | None = None,
        operation_id: str | None = None,
        channel_id: str | None = None,
        activation_timeout_seconds: float = 120.0,
    ) -> list[str]:
        args = [
            str(self.host_python),
            "-m",
            GENERIC_HOST_MODULE,
            "run",
            "--pack-root",
            str(self.pack_root),
            "--runtime-endpoint",
            self.runtime_endpoint,
            "--credential-file",
            str(self.credential_file),
            "--executor-id",
            GENERIC_HOST_EXECUTOR_ID,
            "--ready-file",
            str(self.ready_file),
            "--support-root",
            str(self.support_root),
            "--source-checkout",
            str(self.source_checkout),
            "--runtime-instance-id",
            self.runtime_instance_id,
            "--register",
            "--boot-manifest-path",
            str(self.boot_manifest_path),
            "--boot-manifest-hash",
            self.boot_manifest_hash,
        ]
        if self.capability_matrix is not None:
            args.extend(("--capability-matrix", str(self.capability_matrix)))
        if self.readiness_profile_path is not None and self.readiness_profile_hash is not None:
            args.extend(
                (
                    "--readiness-profile-path",
                    str(self.readiness_profile_path),
                    "--readiness-profile-hash",
                    self.readiness_profile_hash,
                )
            )
        activation_values = (activation_fd, operation_id, channel_id)
        if any(value is not None for value in activation_values):
            if activation_fd is None or not operation_id or not channel_id:
                raise LauncherConfigurationError(
                    "parked host activation fd, operation, and channel must be supplied together"
                )
            args.extend(
                (
                    "--activation-fd",
                    str(activation_fd),
                    "--activation-operation-id",
                    operation_id,
                    "--activation-channel-id",
                    channel_id,
                    "--activation-timeout-seconds",
                    str(float(activation_timeout_seconds)),
                )
            )
        return args


def _host_environment(environ: Mapping[str, str], config: HostLaunchConfig) -> dict[str, str]:
    child_env = {key: value for key, value in environ.items() if key in HOST_ENV_ALLOWLIST}
    # Ambient PYTHONPATH could select another checkout. The configured source
    # checkout is the sole code root admitted to this host.
    child_env["PYTHONPATH"] = str(config.source_checkout)
    child_env["PYTHONUNBUFFERED"] = "1"
    if config.readiness_profile_path is not None and config.readiness_profile_hash is not None:
        child_env["ASTRID_HOST_READINESS_PROFILE_PATH"] = str(config.readiness_profile_path)
        child_env["ASTRID_HOST_READINESS_PROFILE_HASH"] = config.readiness_profile_hash
    return child_env


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(dict(value), sort_keys=True, indent=2), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_returncode(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 + (-returncode)


def _signal_owned_group(process: subprocess.Popen[bytes], signum: int) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _signal_owned_pgid(pgid: int, signum: int) -> None:
    if os.name == "nt":
        return
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def _terminate_and_wait(process: subprocess.Popen[bytes], pgid: int) -> None:
    """Terminate and reap a verified host process group."""

    if os.name == "nt":
        process.terminate()
    else:
        _signal_owned_pgid(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            _signal_owned_pgid(pgid, signal.SIGKILL)
        process.wait(timeout=3)


def _ready_file_is_owned(path: Path, process: subprocess.Popen[bytes]) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("status") == "ready"
        and str(value.get("pid")) == str(process.pid)
    )


def _loopback_endpoint(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port is not None


def _safe_record_path(path: Path, *, support_root: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise LauncherConfigurationError(f"{label} must be an owner-only regular file")
    if path.stat().st_uid != getattr(os, "getuid", lambda: path.stat().st_uid)():
        raise LauncherConfigurationError(f"{label} owner does not match the Worker")
    if path.stat().st_mode & 0o022:
        raise LauncherConfigurationError(f"{label} is writable by another principal")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(support_root):
        raise LauncherConfigurationError(f"{label} must remain beneath the support root")
    return resolved


def _read_runtime_discovery(
    config: HostLaunchConfig,
    *,
    require_worker_credential: bool = True,
) -> RuntimeDiscovery:
    from source.runtime.worker.preflight import _process_birth_identity

    support_root = config.support_root.resolve(strict=True)
    discovery_path = support_root / "discovery.json"
    if discovery_path.is_symlink() or not discovery_path.is_file():
        raise LauncherConfigurationError("canonical Runtime discovery.json is required")
    stat_result = discovery_path.stat()
    if stat_result.st_uid != getattr(os, "getuid", lambda: stat_result.st_uid)() or stat_result.st_mode & 0o022:
        raise LauncherConfigurationError("Runtime discovery.json ownership is unsafe")
    raw = discovery_path.read_bytes()
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherConfigurationError("Runtime discovery.json is malformed") from exc
    allowed = {
        "version", "endpoint", "pid", "process_birth_id", "active_realm", "runtime_instance_id",
        "realm_root", "protocol_version", "schema_version", "coordinator_epoch", "credential_file",
        "worker_credential_file", "worker_actor", "worker_scopes",
    }
    if not isinstance(record, dict) or set(record) != allowed:
        raise LauncherConfigurationError("Runtime discovery.json schema is invalid")
    endpoint = record.get("endpoint")
    parsed = urlsplit(endpoint) if isinstance(endpoint, str) else None
    port = parsed.port if parsed else None
    if record.get("version") != 1 or not parsed or not _loopback_endpoint(endpoint) or port is None:
        raise LauncherConfigurationError("Runtime discovery endpoint or version is invalid")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise LauncherConfigurationError("Runtime discovery endpoint is not canonical")
    if any(not isinstance(record.get(name), str) or not record[name].strip() for name in (
        "process_birth_id", "active_realm", "realm_root", "runtime_instance_id",
        "protocol_version", "schema_version", "coordinator_epoch",
    )):
        raise LauncherConfigurationError("Runtime discovery identity is incomplete")
    realm_root = Path(record["realm_root"])
    if not realm_root.is_absolute() or realm_root.is_symlink() or not realm_root.is_dir():
        raise LauncherConfigurationError("Runtime discovery realm root is invalid")
    try:
        realm_root = realm_root.resolve(strict=True)
    except OSError as exc:
        raise LauncherConfigurationError("Runtime discovery realm root is unavailable") from exc
    if record["protocol_version"] != "workspace.v1" or record["coordinator_epoch"] != record["runtime_instance_id"]:
        raise LauncherConfigurationError("Runtime discovery protocol or coordinator identity is invalid")
    pid = record.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise LauncherConfigurationError("Runtime discovery PID is invalid")
    if _process_birth_identity(pid) != record["process_birth_id"]:
        raise LauncherConfigurationError("Runtime discovery process birth identity is stale")
    worker_actor = record.get("worker_actor")
    worker_scopes = record.get("worker_scopes")
    expected_scopes = {
        "handshake", "worker:register", "worker:execute", "tasks:read", "objects:read", "objects:write",
    }
    if worker_actor != "astrid-pack-host" or not isinstance(worker_scopes, list) or set(worker_scopes) != expected_scopes or len(worker_scopes) != len(expected_scopes):
        raise LauncherConfigurationError("Runtime worker credential scope is invalid")
    worker_raw = record.get("worker_credential_file")
    if not isinstance(worker_raw, str) or not Path(worker_raw).is_absolute():
        raise LauncherConfigurationError("Runtime worker credential reference is invalid")
    worker_candidate = Path(worker_raw)
    if require_worker_credential:
        worker_path = _safe_record_path(
            worker_candidate,
            support_root=support_root,
            label="Runtime worker credential",
        )
        if worker_path.stat().st_mode & 0o777 != 0o600:
            raise LauncherConfigurationError("Runtime worker credential must be owner-only")
    else:
        try:
            worker_parent = worker_candidate.parent.resolve(strict=True)
        except OSError as exc:
            raise LauncherConfigurationError(
                "Runtime worker credential parent is unavailable"
            ) from exc
        worker_path = worker_parent / worker_candidate.name
        if not worker_path.is_relative_to(support_root):
            raise LauncherConfigurationError(
                "Runtime worker credential reference must remain beneath the support root"
            )
        if worker_candidate.exists() or worker_candidate.is_symlink():
            worker_path = _safe_record_path(
                worker_candidate,
                support_root=support_root,
                label="Runtime worker credential",
            )
            if worker_path.stat().st_mode & 0o777 != 0o600:
                raise LauncherConfigurationError(
                    "Runtime worker credential must be owner-only"
                )
    if config.runtime_endpoint.rstrip("/") != endpoint.rstrip("/"):
        raise LauncherConfigurationError("configured Runtime endpoint conflicts with discovery")
    if config.runtime_instance_id != record["runtime_instance_id"]:
        raise LauncherConfigurationError("configured Runtime instance conflicts with discovery")
    try:
        configured_credential = (
            config.credential_file.resolve(strict=True)
            if require_worker_credential
            else config.credential_file.parent.resolve(strict=True) / config.credential_file.name
        )
    except OSError as exc:
        raise LauncherConfigurationError("configured Runtime credential is unavailable") from exc
    if configured_credential != worker_path:
        raise LauncherConfigurationError("configured credential is not the scoped Worker credential")
    return RuntimeDiscovery(
        endpoint=endpoint.rstrip("/"), port=port, pid=pid, process_birth_id=record["process_birth_id"],
        runtime_instance_id=record["runtime_instance_id"], coordinator_epoch=record["coordinator_epoch"],
        active_realm=record["active_realm"], realm_root=realm_root,
        protocol_version=record["protocol_version"], schema_version=record["schema_version"],
        worker_credential_file=worker_path, worker_actor=worker_actor, worker_scopes=tuple(worker_scopes),
        snapshot_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _prepare_worker_readiness(
    config: HostLaunchConfig,
    environ: Mapping[str, str],
    *,
    vibecomfy_session: dict[str, object] | None = None,
) -> tuple[Path, str]:
    """Bind discovery, verify neutral facts, and publish one HC-03 profile."""

    from source.runtime.worker.preflight import (
        RuntimeBinding,
        _probe_runtime_binding,
        _read_runtime_health,
        _verify_runtime_process,
        run_neutral_worker_preflight,
    )

    profile_path = config.support_root / "worker-readiness-profile.json"
    profile_path.unlink(missing_ok=True)
    discovery = _read_runtime_discovery(config)
    provisional = RuntimeBinding(
        endpoint=discovery.endpoint, port=discovery.port, pid=discovery.pid,
        process_birth_id=discovery.process_birth_id, runtime_instance_id=discovery.runtime_instance_id,
        runtime_epoch=1, schema_digest="sha256:" + "0" * 64, credential_path=discovery.worker_credential_file,
    )
    _verify_runtime_process(provisional)
    health = _read_runtime_health(provisional)
    if health.get("status") != "ok" or health.get("protocol") != "workspace.v1":
        raise LauncherConfigurationError("Runtime health status or protocol conflicts with discovery")
    if not isinstance(health.get("schema_digest"), str) or not isinstance(health.get("runtime_epoch"), int):
        raise LauncherConfigurationError("Runtime health identity is incomplete")
    binding = RuntimeBinding(
        endpoint=discovery.endpoint, port=discovery.port, pid=discovery.pid,
        process_birth_id=discovery.process_birth_id, runtime_instance_id=discovery.runtime_instance_id,
        runtime_epoch=health["runtime_epoch"], schema_digest=health["schema_digest"], credential_path=discovery.worker_credential_file,
    )
    _probe_runtime_binding(binding)
    if _read_runtime_discovery(config).snapshot_digest != discovery.snapshot_digest:
        raise LauncherConfigurationError("Runtime discovery changed during readiness verification")
    fact_inputs = {
        name: environ.get(name, "")
        for name in (
            "REIGH_INTERPRETER", "REIGH_ENGINE_INTERPRETER", "REIGH_RUNTIME_LOCK_PATH", "REIGH_ENGINE_LOCK_PATH",
            "REIGH_MODEL_ROOT", "REIGH_MODEL_MANIFEST_PATH", "REIGH_CUSTOM_NODE_ROOT", "REIGH_CUSTOM_NODE_MANIFEST_PATH",
            "REIGH_SCRATCH_ROOT", "REIGH_CAS_ROOT", "REIGH_OUTPUT_ROOT",
        )
    }
    result = run_neutral_worker_preflight(
        repo_root=config.source_checkout,
        main_output_dir=Path(fact_inputs.get("REIGH_OUTPUT_ROOT") or (config.support_root / "outputs")),
        fact_inputs=fact_inputs,
        runtime_binding=binding,
    )
    if not result.ready_for_tasks:
        raise LauncherConfigurationError("neutral Worker readiness facts are incomplete or failed")
    metadata = result.to_metadata()
    payload = {
        "schema_version": "hc03-worker-readiness.v1",
        "status": "ready",
        "verified_facts": metadata["verified_facts"],
        "verified_facts_digest": metadata["verified_facts_digest"],
        "runtime": {
            "endpoint": binding.endpoint, "port": binding.port, "pid": binding.pid,
            "process_birth_id": binding.process_birth_id, "runtime_instance_id": binding.runtime_instance_id,
            "runtime_epoch": binding.runtime_epoch, "schema_digest": binding.schema_digest,
            "coordinator_epoch": discovery.coordinator_epoch, "active_realm": discovery.active_realm,
            "credential_reference": str(binding.credential_path), "discovery_digest": discovery.snapshot_digest,
        },
        "launch": {
            "host_interpreter": str(config.host_python), "source_checkout": str(config.source_checkout),
            "engine_interpreter": fact_inputs.get("REIGH_ENGINE_INTERPRETER"),
            "output_root": fact_inputs.get("REIGH_OUTPUT_ROOT"), "pack_root": str(config.pack_root), "support_root": str(config.support_root),
            "ready_file": str(config.ready_file), "state_file": str(config.state_file),
            "boot_manifest_path": str(config.boot_manifest_path), "boot_manifest_hash": config.boot_manifest_hash,
        },
        "worker_actor": discovery.worker_actor, "worker_scopes": list(discovery.worker_scopes),
    }
    if vibecomfy_session is not None:
        payload["vibecomfy_session"] = vibecomfy_session
    try:
        _atomic_write_json(profile_path, payload)
        profile_hash = "sha256:" + hashlib.sha256(profile_path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError) as exc:
        profile_path.unlink(missing_ok=True)
        raise LauncherConfigurationError("HC-03 readiness profile publication failed") from exc
    return profile_path, profile_hash


def _process_parent_pid(pid: int) -> int | None:
    """Return a process parent identity for the composite ownership proof."""
    try:
        result = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        value = result.stdout.strip()
        return int(value) if result.returncode == 0 and value else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _read_owned_vibecomfy_session(
    root: Path,
    *,
    expected_daemon_pid: int | None = None,
    verify_parent: bool = True,
    require_attestation: bool = True,
) -> dict[str, object]:
    """Read a registry produced by this Worker-owned session launch."""
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise LauncherConfigurationError(
            "Worker-owned VibeComfy session directory must be an existing non-symlink directory"
        )
    registry = {
        name: root / name for name in (
            "pid", "comfy_pid", "comfy_process_start_identity", "url",
            "config.json", "source_revision", "source_content_digest",
            "launch.json", "daemon.log",
        )
    }
    required_registry = tuple(registry.values())
    if not require_attestation:
        required_registry = tuple(
            path for name, path in registry.items() if name != "source_content_digest"
        )
    if any(not path.is_file() or path.is_symlink() for path in required_registry):
        raise LauncherConfigurationError("VibeComfy session registry is incomplete")
    try:
        pid = int(registry["pid"].read_text(encoding="utf-8").strip())
        if pid <= 0:
            raise ValueError("pid must be positive")
        if expected_daemon_pid is not None and pid != expected_daemon_pid:
            raise ValueError("session daemon pid does not match the Worker-owned child")
        os.kill(pid, 0)
        comfy_pid = int(registry["comfy_pid"].read_text(encoding="utf-8").strip())
        if comfy_pid <= 0:
            raise ValueError("comfy_pid must be positive")
        from source.runtime.worker.preflight import _process_birth_identity
        comfy_process_birth_id = registry["comfy_process_start_identity"].read_text(encoding="utf-8").strip()
        if _process_birth_identity(comfy_pid) != comfy_process_birth_id:
            raise ValueError("Comfy child process birth identity is stale or mismatched")
        url = registry["url"].read_text(encoding="utf-8").strip()
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port is None:
            raise ValueError("VibeComfy session URL must be loopback HTTP")
        listener = subprocess.run(
            ["lsof", "-nP", "-a", "-p", str(comfy_pid), "-iTCP:" + str(parsed.port), "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        listener_stderr = getattr(listener, "stderr", "") or ""
        if listener.returncode == 1 and not listener.stdout.strip() and not listener_stderr.strip():
            raise ValueError("VibeComfy listener is absent")
        if listener.returncode != 0 or f":{parsed.port} (LISTEN)" not in listener.stdout:
            raise ValueError("VibeComfy listener is not owned by the recorded Comfy child")
        if verify_parent and _process_parent_pid(comfy_pid) != pid:
            raise ValueError("VibeComfy Comfy child is not parented by the owned daemon")
        source_revision = registry["source_revision"].read_text(encoding="utf-8").strip()
        source_content_digest = ""
        if registry["source_content_digest"].is_file():
            source_content_digest = registry["source_content_digest"].read_text(encoding="utf-8").strip()
        marker = json.loads(registry["launch.json"].read_text(encoding="utf-8"))
        config_digest = "sha256:" + hashlib.sha256(registry["config.json"].read_bytes()).hexdigest()
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise LauncherConfigurationError("VibeComfy session registry is unreadable or stale") from exc
    if not isinstance(marker, Mapping):
        raise LauncherConfigurationError("VibeComfy launch marker is malformed")
    launch_token = marker.get("launch_token")
    process_birth_id = marker.get("process_start_identity")
    if (
        marker.get("pid") != pid
        or marker.get("url") != url
        or not isinstance(launch_token, str)
        or not launch_token.strip()
        or not isinstance(process_birth_id, str)
        or not process_birth_id.strip()
        or not source_revision
        or (
            require_attestation
            and (
                not source_content_digest.startswith("sha256:")
                or len(source_content_digest) != len("sha256:") + 64
            )
        )
    ):
        raise LauncherConfigurationError("VibeComfy launch marker does not bind the registry")
    from source.runtime.worker.preflight import _process_birth_identity
    if _process_birth_identity(pid) != process_birth_id:
        raise LauncherConfigurationError("VibeComfy daemon process birth identity is stale or mismatched")
    return {
        "session_dir": str(root),
        "server_url": url,
        "pid": pid,
        "comfy_pid": comfy_pid,
        "comfy_process_birth_id": comfy_process_birth_id,
        "launch_token": launch_token,
        "process_birth_id": process_birth_id,
        "source_revision": source_revision,
        "source_content_digest": source_content_digest,
        "config_digest": config_digest,
    }


def _read_owned_vibecomfy_custody(
    root: Path,
    *,
    expected_daemon_pid: int,
    expected_server_url: str,
) -> dict[str, object] | None:
    """Recover launch custody without requiring readiness or liveness.

    This is intentionally weaker than ``_read_owned_vibecomfy_session``.  It
    is used only while unwinding a failed launch, when the daemon may already
    have exited and Vibe may not yet have published all readiness files.
    """
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        return None
    pid_path = root / "pid"
    if not pid_path.is_file() or pid_path.is_symlink():
        return None
    try:
        daemon_pid = int(pid_path.read_text(encoding="utf-8").strip())
        if daemon_pid != expected_daemon_pid or daemon_pid <= 0:
            return None
        url = expected_server_url
        url_path = root / "url"
        if url_path.is_file() and not url_path.is_symlink():
            candidate_url = url_path.read_text(encoding="utf-8").strip()
            parsed = urlsplit(candidate_url)
            if (
                parsed.scheme == "http"
                and parsed.hostname in {"127.0.0.1", "localhost"}
                and parsed.port is not None
            ):
                url = candidate_url
        comfy_pid_path = root / "comfy_pid"
        birth_path = root / "comfy_process_start_identity"
        comfy_pid = 0
        comfy_birth = ""
        if comfy_pid_path.is_file() and birth_path.is_file():
            comfy_pid = int(comfy_pid_path.read_text(encoding="utf-8").strip())
            comfy_birth = birth_path.read_text(encoding="utf-8").strip()
            if comfy_pid <= 0 or not comfy_birth:
                return None
    except (OSError, UnicodeError, ValueError):
        return None
    return {
        "pid": daemon_pid,
        "comfy_pid": comfy_pid,
        "comfy_process_birth_id": comfy_birth,
        "server_url": url,
    }


def _assert_owned_vibecomfy_port_available(port: int) -> None:
    """Verify that the configured port is unused immediately before launch."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", "-iTCP:" + str(port), "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LauncherConfigurationError(
            "owned VibeComfy port availability could not be verified"
        ) from exc
    stderr = getattr(result, "stderr", "") or ""
    if result.returncode == 1 and not result.stdout.strip() and not stderr.strip():
        return
    if result.returncode == 0 and result.stdout.strip():
        raise LauncherConfigurationError(
            "owned VibeComfy launch port is already in use"
        )
    raise LauncherConfigurationError(
        "owned VibeComfy port availability could not be verified"
    )


def _owned_listener_pid(port: int) -> int | None:
    """Return the sole listener PID, or fail closed on probe errors."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", "-iTCP:" + str(port), "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LauncherConfigurationError(
            "owned VibeComfy listener could not be observed"
        ) from exc
    stderr = getattr(result, "stderr", "") or ""
    if result.returncode == 1 and not result.stdout.strip() and not stderr.strip():
        return None
    if result.returncode != 0 or not result.stdout.strip():
        raise LauncherConfigurationError(
            "owned VibeComfy listener could not be observed"
        )
    pids = result.stdout.split()
    if len(pids) != 1:
        raise LauncherConfigurationError(
            "owned VibeComfy listener ownership is ambiguous"
        )
    try:
        return int(pids[0])
    except ValueError as exc:
        raise LauncherConfigurationError(
            "owned VibeComfy listener PID is invalid"
        ) from exc


@dataclass(frozen=True)
class _OwnedVibeComfySession:
    root: Path
    process: subprocess.Popen[bytes]
    daemon_pid: int = 0
    comfy_pid: int = 0
    comfy_process_birth_id: str = ""
    server_url: str = ""

def _start_owned_vibecomfy_session(
    config: HostLaunchConfig,
    environ: Mapping[str, str],
) -> tuple[dict[str, object] | None, _OwnedVibeComfySession | None]:
    """Start VibeComfy under Worker custody before publishing readiness.

    A configured session directory is a launch target, never an adoption
    source.  Any pre-existing registry is rejected; only the daemon spawned by
    this call may publish the HC-03 Vibe extension.
    """
    raw_root = environ.get("ASTRID_VIBECOMFY_SESSION_DIR", "").strip()
    if not raw_root:
        return None, None
    root = Path(raw_root)
    if not root.is_absolute() or root.is_symlink():
        raise LauncherConfigurationError(
            "ASTRID_VIBECOMFY_SESSION_DIR must be an absolute non-symlink path"
        )
    if root.parent.name != "sessions" or root.parent.parent.name != "out":
        raise LauncherConfigurationError(
            "ASTRID_VIBECOMFY_SESSION_DIR must use the VibeComfy out/sessions/<id> layout"
        )
    root.mkdir(parents=True, exist_ok=True)
    registry_names = (
        "pid", "comfy_pid", "comfy_process_start_identity", "url",
        "config.json", "source_revision", "source_content_digest",
        "launch.json", "daemon.log",
    )
    if any((root / name).exists() or (root / name).is_symlink() for name in registry_names):
        raise LauncherConfigurationError(
            "pre-existing VibeComfy session registry cannot be adopted; use a fresh owned session directory"
        )
    try:
        config_values: dict[str, object] = {}
        raw_config = environ.get("ASTRID_VIBECOMFY_SESSION_CONFIG", "").strip()
        if raw_config:
            parsed = json.loads(raw_config)
            if not isinstance(parsed, dict):
                raise ValueError("ASTRID_VIBECOMFY_SESSION_CONFIG must be an object")
            config_values = dict(parsed)
        comfyui_path = environ.get("COMFYUI_PATH", "").strip()
        comfyui_root: Path | None = None
        if comfyui_path:
            candidate = Path(comfyui_path).expanduser()
            if (
                not candidate.is_absolute()
                or candidate.is_symlink()
                or not candidate.is_dir()
            ):
                raise ValueError("COMFYUI_PATH must be an absolute non-symlink directory")
            comfyui_root = candidate.resolve(strict=True)
            config_values["base_directory"] = str(comfyui_root)
            extra_model_paths = comfyui_root / "extra_model_paths.yaml"
            if extra_model_paths.is_file() and not extra_model_paths.is_symlink():
                config_values["extra_model_paths_config"] = [
                    str(extra_model_paths.resolve())
                ]
        port = int(environ.get("ASTRID_VIBECOMFY_PORT", "8188"))
        if not 1 <= port <= 65535:
            raise ValueError("ASTRID_VIBECOMFY_PORT is outside the valid range")
        timeout = float(environ.get("VIBECOMFY_SESSION_READY_TIMEOUT_SEC", "300"))
        if not timeout > 0:
            raise ValueError("VIBECOMFY_SESSION_READY_TIMEOUT_SEC must be positive")
        config_values.update(
            {
                "port": port,
                "warm_policy": "auto",
                "locality": "managed_local_server",
                "runtime_root": str(root.parents[2]),
                "cwd": str(root.parents[2]),
                "server_log_path": str(root / "comfy.log"),
                "ready_timeout_sec": timeout,
            }
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LauncherConfigurationError("owned VibeComfy session configuration is invalid") from exc

    expected_server_url = f"http://127.0.0.1:{port}"
    _assert_owned_vibecomfy_port_available(port)

    interpreter = Path(
        environ.get("REIGH_ENGINE_INTERPRETER", "").strip() or sys.executable
    ).expanduser()
    if not interpreter.is_absolute() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise LauncherConfigurationError("owned VibeComfy session interpreter is unavailable")
    launch_token = uuid.uuid4().hex
    command = [
        str(interpreter),
        "-m",
        "vibecomfy.commands.session",
        "--daemon",
        "--id",
        root.name,
        "--require-source-attestation",
        "--launch-token",
        launch_token,
        "--config",
        json.dumps(config_values, sort_keys=True, separators=(",", ":")),
    ]
    child_env = {
        key: value
        for key, value in environ.items()
        if key in {
            "PATH", "LANG", "LC_ALL", "LC_CTYPE", "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        }
    }
    if comfyui_root is not None:
        child_env["COMFYUI_PATH"] = str(comfyui_root)
    child_env["PYTHONUNBUFFERED"] = "1"
    log_path = root / "daemon.log"
    try:
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            # VibeComfy resolves its registry as cwd/out/sessions/<id>.
            # For <base>/out/sessions/<id>, cwd must therefore be <base>.
            cwd=str(root.parents[2]),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log_handle,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LauncherConfigurationError("owned VibeComfy session could not be started") from exc
    finally:
        try:
            log_handle.close()
        except UnboundLocalError:
            pass

    deadline = time.monotonic() + max(1.0, min(timeout, 900.0))
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise LauncherConfigurationError("owned VibeComfy session daemon exited before readiness")
            try:
                payload = _read_owned_vibecomfy_session(
                    root,
                    expected_daemon_pid=process.pid,
                    verify_parent=True,
                )
                return payload, _OwnedVibeComfySession(
                    root=root,
                    process=process,
                    daemon_pid=int(payload["pid"]),
                    comfy_pid=int(payload["comfy_pid"]),
                    comfy_process_birth_id=str(payload["comfy_process_birth_id"]),
                    server_url=str(payload["server_url"]),
                )
            except LauncherConfigurationError:
                time.sleep(0.1)
        raise LauncherConfigurationError("owned VibeComfy session did not become ready")
    except BaseException:
        cleanup_session = _OwnedVibeComfySession(
            root=root,
            process=process,
            daemon_pid=process.pid,
            server_url=expected_server_url,
        )
        partial = _read_owned_vibecomfy_custody(
            root,
            expected_daemon_pid=process.pid,
            expected_server_url=expected_server_url,
        )
        if partial is not None:
            cleanup_session = _OwnedVibeComfySession(
                root=root,
                process=process,
                daemon_pid=int(partial["pid"]),
                comfy_pid=int(partial["comfy_pid"]),
                comfy_process_birth_id=str(partial["comfy_process_birth_id"]),
                server_url=str(partial["server_url"]),
            )
        _stop_owned_vibecomfy_session(cleanup_session)
        raise


def _stop_owned_vibecomfy_session(session: _OwnedVibeComfySession) -> None:
    process = session.process
    if process.poll() is None:
        _signal_owned_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _signal_owned_group(process, signal.SIGKILL)
            process.wait(timeout=15)

    from source.runtime.worker.preflight import _process_birth_identity

    child_pid = session.comfy_pid
    child_birth_id = session.comfy_process_birth_id
    parsed = urlsplit(session.server_url)
    def child_state() -> str:
        """Return owned, absent, or unknown without adopting a PID."""
        if child_pid <= 0 or not child_birth_id:
            return "unknown"
        observed_birth_id = _process_birth_identity(child_pid)
        if observed_birth_id is None:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return "absent"
            except OSError:
                return "unknown"
            return "unknown"
        if observed_birth_id != child_birth_id:
            return "absent"
        if os.name == "nt":
            return "owned"
        try:
            return "owned" if os.getpgid(child_pid) == session.process.pid else "unknown"
        except OSError:
            return "unknown"

    if child_state() == "owned":
        try:
            os.kill(child_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 15
        while child_state() == "owned" and time.monotonic() < deadline:
            time.sleep(0.1)
        if child_state() == "owned":
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 15
            while child_state() == "owned" and time.monotonic() < deadline:
                time.sleep(0.1)

    if parsed.port is not None:
        listener_pid = _owned_listener_pid(parsed.port)
        if listener_pid is None:
            if child_state() != "absent":
                raise LauncherConfigurationError(
                    "owned VibeComfy child absence could not be verified"
                )
            return
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            listener_pid = _owned_listener_pid(parsed.port)
            if listener_pid is None:
                if child_state() != "absent":
                    raise LauncherConfigurationError(
                        "owned VibeComfy child absence could not be verified"
                    )
                return
            time.sleep(0.1)
        raise LauncherConfigurationError(
            "owned VibeComfy listener remained after cleanup"
        )


@dataclass
class PreparedHostHandle:
    """Worker-owned engine, parked host, and inherited activation endpoint."""

    profile: object
    operation_id: str
    channel_id: str
    host: subprocess.Popen[bytes]
    host_birth_id: str
    activation: socket.socket
    engine: _OwnedVibeComfySession
    engine_report: dict[str, object]
    readiness_profile: Path | None = None
    activated: bool = False
    activation_grant: dict[str, object] | None = None
    closed: bool = False


class LocalWorkerPreparerAdapter:
    """Concrete Worker-side implementation of Runtime's private preparer ABI.

    Runtime remains the issuer and independent observer.  This object owns only
    process preparation and the one inherited Worker-to-host activation socket.
    A transport proxy may invoke these methods in the Worker process; no grant,
    credential, or placement assertion is accepted from ambient environment.
    """

    def __init__(
        self,
        config: HostLaunchConfig,
        *,
        environ: Mapping[str, str] | None = None,
        activation_timeout_seconds: float = 120.0,
    ):
        self.config = config
        self.environ = dict(os.environ if environ is None else environ)
        self.activation_timeout_seconds = float(activation_timeout_seconds)
        self._active: PreparedHostHandle | None = None

    @staticmethod
    def _birth(pid: int) -> str:
        from source.runtime.worker.preflight import _process_birth_identity

        birth = _process_birth_identity(pid)
        if not birth:
            raise LauncherConfigurationError("prepared process birth identity is unavailable")
        return birth

    @staticmethod
    def _profile_value(profile: object, name: str) -> object:
        try:
            return getattr(profile, name)
        except AttributeError as exc:
            raise LauncherConfigurationError(
                f"local Worker profile is missing {name}"
            ) from exc

    def _validate_profile(self, profile: object) -> RuntimeDiscovery:
        discovery = _read_runtime_discovery(
            self.config, require_worker_credential=False
        )
        if str(self._profile_value(profile, "workspace_uuid")) != discovery.active_realm:
            raise LauncherConfigurationError("local Worker profile workspace identity is invalid")
        if Path(self._profile_value(profile, "realm_root")).resolve() != discovery.realm_root:
            raise LauncherConfigurationError("local Worker profile realm root is invalid")
        if Path(self._profile_value(profile, "support_root")).resolve() != self.config.support_root.resolve():
            raise LauncherConfigurationError("local Worker profile support root is invalid")
        if Path(self._profile_value(profile, "host_executable")).resolve() != self.config.host_python.resolve():
            raise LauncherConfigurationError("local Worker profile host executable is invalid")
        worker_executable = Path(self._profile_value(profile, "worker_executable"))
        if worker_executable.resolve() != Path(sys.executable).resolve():
            raise LauncherConfigurationError("local Worker profile worker executable is invalid")
        return discovery

    def _assert_live(self, handle: PreparedHostHandle, *, parked: bool = False) -> None:
        if handle.closed or handle.host.poll() is not None:
            raise LauncherConfigurationError("prepared GenericPackHost is not alive")
        if self._birth(handle.host.pid) != handle.host_birth_id:
            raise LauncherConfigurationError("prepared GenericPackHost identity changed")
        if os.name != "nt":
            try:
                if os.getpgid(handle.host.pid) != handle.host.pid or os.getsid(handle.host.pid) != handle.host.pid:
                    raise LauncherConfigurationError(
                        "prepared GenericPackHost lost process-group or session custody"
                    )
            except OSError as exc:
                raise LauncherConfigurationError(
                    "prepared GenericPackHost custody is unavailable"
                ) from exc
        current = _read_owned_vibecomfy_session(
            handle.engine.root,
            expected_daemon_pid=handle.engine.daemon_pid,
            verify_parent=True,
        )
        for key in (
            "pid", "process_birth_id", "comfy_pid", "comfy_process_birth_id",
            "server_url", "config_digest",
        ):
            if current.get(key) != handle.engine_report.get(key):
                raise LauncherConfigurationError("prepared VibeComfy identity changed")
        if parked and self.config.ready_file.exists():
            raise LauncherConfigurationError("parked GenericPackHost published readiness before activation")

    def prepare(
        self,
        profile: object,
        *,
        operation_id: str,
        channel_id: str,
    ) -> PreparedHostHandle:
        if not operation_id or not channel_id:
            raise LauncherConfigurationError("prepared operation and channel identities are required")
        if self._active is not None:
            self.abort(self._active)
        self._validate_profile(profile)
        engine_report, engine = _start_owned_vibecomfy_session(
            self.config, self.environ
        )
        if engine_report is None or engine is None:
            raise LauncherConfigurationError("prepared local Worker requires an owned VibeComfy session")
        expected_config_digest = str(self._profile_value(profile, "session_config_digest"))
        if engine_report.get("config_digest") != expected_config_digest:
            _stop_owned_vibecomfy_session(engine)
            raise LauncherConfigurationError("prepared VibeComfy session configuration is invalid")
        parent_control, host_control = socket.socketpair()
        self.config.ready_file.unlink(missing_ok=True)
        child: subprocess.Popen[bytes] | None = None
        try:
            argv = self.config.argv(
                activation_fd=host_control.fileno(),
                operation_id=operation_id,
                channel_id=channel_id,
                activation_timeout_seconds=self.activation_timeout_seconds,
            )
            child = subprocess.Popen(
                argv,
                cwd=str(self.config.source_checkout),
                env=_host_environment(self.environ, self.config),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                pass_fds=(host_control.fileno(),),
            )
            host_control.close()
            birth = self._birth(child.pid)
            handle = PreparedHostHandle(
                profile=profile,
                operation_id=operation_id,
                channel_id=channel_id,
                host=child,
                host_birth_id=birth,
                activation=parent_control,
                engine=engine,
                engine_report=dict(engine_report),
            )
            self._assert_live(handle, parked=True)
            self._active = handle
            return handle
        except BaseException:
            host_control.close()
            parent_control.close()
            if child is not None:
                try:
                    _terminate_and_wait(child, child.pid)
                except BaseException:
                    pass
            _stop_owned_vibecomfy_session(engine)
            raise

    def report(self, handle: object) -> Mapping[str, Any]:
        if not isinstance(handle, PreparedHostHandle) or handle is not self._active:
            raise LauncherConfigurationError("prepared host handle is not owned by this Worker")
        self._assert_live(handle, parked=not handle.activated)
        worker_birth = self._birth(os.getpid())
        return {
            "version": PREPARATION_VERSION,
            "operation_id": handle.operation_id,
            "channel_id": handle.channel_id,
            "processes": {
                "worker": {"pid": os.getpid(), "birth_id": worker_birth},
                "host": {"pid": handle.host.pid, "birth_id": handle.host_birth_id},
                "engine": {
                    "pid": handle.engine.daemon_pid,
                    "birth_id": str(handle.engine_report["process_birth_id"]),
                },
                "engine_listener": {
                    "pid": handle.engine.comfy_pid,
                    "birth_id": handle.engine.comfy_process_birth_id,
                },
            },
            "engine_binding": {
                "supervisor_pid": handle.engine.daemon_pid,
                "listener_pid": handle.engine.comfy_pid,
                "listener_parent_pid": handle.engine.daemon_pid,
                "socket_owner_pid": handle.engine.comfy_pid,
            },
            "session_config_digest": str(handle.engine_report["config_digest"]),
        }

    def activate(self, handle: object, grant: Mapping[str, Any]) -> None:
        if not isinstance(handle, PreparedHostHandle) or handle is not self._active:
            raise LauncherConfigurationError("prepared host handle is not owned by this Worker")
        if handle.activated:
            raise LauncherConfigurationError("prepared host has already consumed an activation grant")
        self._assert_live(handle, parked=True)
        required = {
            "version", "operation_id", "channel_id", "credential_file",
            "executor_incarnation", "evidence_digest",
        }
        if not isinstance(grant, Mapping) or set(grant) != required:
            raise LauncherConfigurationError("Runtime activation grant has an invalid shape")
        if (
            grant.get("version") != ACTIVATION_VERSION
            or grant.get("operation_id") != handle.operation_id
            or grant.get("channel_id") != handle.channel_id
        ):
            raise LauncherConfigurationError("Runtime activation grant is stale or on the wrong channel")
        credential = Path(str(grant.get("credential_file", "")))
        if credential != self.config.credential_file or credential.is_symlink() or not credential.is_file():
            raise LauncherConfigurationError("Runtime activation credential reference is invalid")
        if credential.stat().st_mode & 0o777 != 0o600:
            raise LauncherConfigurationError("Runtime activation credential is not owner-only")
        incarnation = grant.get("executor_incarnation")
        digest = grant.get("evidence_digest")
        if not isinstance(incarnation, str) or not incarnation or len(incarnation) > 256:
            raise LauncherConfigurationError("Runtime activation incarnation is invalid")
        if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
            raise LauncherConfigurationError("Runtime activation evidence digest is invalid")
        wire = {
            **dict(grant),
            "host": {"pid": handle.host.pid, "birth_id": handle.host_birth_id},
        }
        handle.activation.settimeout(self.activation_timeout_seconds)
        handle.activation.sendall(
            json.dumps(wire, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        frame = bytearray()
        while b"\n" not in frame:
            chunk = handle.activation.recv(min(4096, _CONTROL_FRAME_LIMIT + 1 - len(frame)))
            if not chunk:
                raise LauncherConfigurationError("parked GenericPackHost rejected activation")
            frame.extend(chunk)
            if len(frame) > _CONTROL_FRAME_LIMIT:
                raise LauncherConfigurationError("parked GenericPackHost activation response is too large")
        try:
            accepted = json.loads(bytes(frame).split(b"\n", 1)[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LauncherConfigurationError("parked GenericPackHost activation response is malformed") from exc
        expected_ack = {
            "version": ACTIVATION_ACCEPTED_VERSION,
            "operation_id": handle.operation_id,
            "channel_id": handle.channel_id,
            "executor_incarnation": incarnation,
            "evidence_digest": digest,
            "host": wire["host"],
        }
        if accepted != expected_ack:
            raise LauncherConfigurationError("parked GenericPackHost activation response is invalid")
        handle.activation.close()
        handle.activated = True
        handle.activation_grant = dict(grant)

    def abort(self, handle: object) -> None:
        if not isinstance(handle, PreparedHostHandle) or handle is not self._active:
            return
        error: BaseException | None = None
        try:
            if handle.activation.fileno() >= 0:
                handle.activation.close()
            if handle.host.poll() is None:
                if self._birth(handle.host.pid) != handle.host_birth_id:
                    raise LauncherConfigurationError(
                        "prepared host replacement prevents safe cleanup"
                    )
                _terminate_and_wait(handle.host, handle.host.pid)
            _stop_owned_vibecomfy_session(handle.engine)
        except BaseException as exc:
            error = exc
        finally:
            self.config.ready_file.unlink(missing_ok=True)
            if handle.readiness_profile is not None:
                handle.readiness_profile.unlink(missing_ok=True)
            handle.closed = True
            self._active = None
        if error is not None:
            raise error

    def reconnect(self, receipt: Mapping[str, Any]) -> object | None:
        handle = self._active
        if handle is None or handle.closed or not handle.activated:
            return None
        try:
            report = self.report(handle)
        except LauncherConfigurationError:
            self.abort(handle)
            raise
        if receipt.get("version") != RECEIPT_VERSION:
            return None
        processes = (
            receipt.get("worker"),
            receipt.get("host"),
            receipt.get("engine"),
            receipt.get("engine_listener"),
        )
        names = ("worker", "host", "engine", "engine_listener")
        for name, process in zip(names, processes):
            expected = report["processes"][name]
            if not isinstance(process, Mapping) or any(process.get(key) != expected[key] for key in ("pid", "birth_id")):
                return None
        grant = handle.activation_grant or {}
        if (
            receipt.get("session_config_digest") != report["session_config_digest"]
            or receipt.get("evidence_digest") != grant.get("evidence_digest")
            or receipt.get("executor_incarnation") != grant.get("executor_incarnation")
        ):
            return None
        return handle


@dataclass
class PreparedWorkerHandle:
    """Runtime-side custody of the pinned Worker and its private control fd."""

    worker: subprocess.Popen[bytes]
    worker_birth_id: str
    control: socket.socket
    report_value: dict[str, Any]
    activated: bool = False
    closed: bool = False


def _send_private_frame(channel: socket.socket, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > _CONTROL_FRAME_LIMIT:
        raise LauncherConfigurationError("local Worker control frame is too large")
    channel.sendall(encoded + b"\n")


def _receive_private_frame(channel: socket.socket) -> dict[str, Any]:
    frame = bytearray()
    while b"\n" not in frame:
        chunk = channel.recv(min(4096, _CONTROL_FRAME_LIMIT + 1 - len(frame)))
        if not chunk:
            raise LauncherConfigurationError("local Worker control channel closed")
        frame.extend(chunk)
        if len(frame) > _CONTROL_FRAME_LIMIT:
            raise LauncherConfigurationError("local Worker control frame is too large")
    encoded, remainder = bytes(frame).split(b"\n", 1)
    if remainder:
        raise LauncherConfigurationError("local Worker control channel carried multiple frames")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherConfigurationError("local Worker control frame is malformed") from exc
    if not isinstance(value, dict):
        raise LauncherConfigurationError("local Worker control frame must be an object")
    return value


_PROFILE_FIELDS = (
    "profile_id", "workspace_uuid", "realm_root", "support_root", "machine_id",
    "worker_executable", "host_executable", "engine_executable",
    "engine_listener_executable", "worker_artifact_digest", "host_artifact_digest",
    "engine_artifact_digest", "engine_listener_artifact_digest",
    "session_config_digest", "profile_revision", "profile_digest", "release_digest",
)


def _private_profile_payload(profile: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _PROFILE_FIELDS:
        try:
            value = getattr(profile, name)
        except AttributeError as exc:
            raise LauncherConfigurationError(f"local Worker profile is missing {name}") from exc
        result[name] = str(value) if isinstance(value, Path) else value
    return result


def _private_config_payload(config: HostLaunchConfig) -> dict[str, Any]:
    return {
        "host_python": str(config.host_python),
        "source_checkout": str(config.source_checkout),
        "pack_root": str(config.pack_root),
        "runtime_endpoint": config.runtime_endpoint,
        "credential_file": str(config.credential_file),
        "support_root": str(config.support_root),
        "runtime_instance_id": config.runtime_instance_id,
        "ready_file": str(config.ready_file),
        "state_file": str(config.state_file),
        "boot_manifest_path": str(config.boot_manifest_path),
        "boot_manifest_hash": config.boot_manifest_hash,
        "capability_matrix": str(config.capability_matrix) if config.capability_matrix else None,
        "readiness_profile_path": str(config.readiness_profile_path) if config.readiness_profile_path else None,
        "readiness_profile_hash": config.readiness_profile_hash,
    }


def _config_from_private_payload(value: Mapping[str, Any]) -> HostLaunchConfig:
    expected = {
        "host_python", "source_checkout", "pack_root", "runtime_endpoint",
        "credential_file", "support_root", "runtime_instance_id", "ready_file",
        "state_file", "boot_manifest_path", "boot_manifest_hash", "capability_matrix",
        "readiness_profile_path", "readiness_profile_hash",
    }
    if set(value) != expected:
        raise LauncherConfigurationError("private host configuration has an invalid shape")
    path_fields = {
        "host_python", "source_checkout", "pack_root", "credential_file", "support_root",
        "ready_file", "state_file", "boot_manifest_path", "capability_matrix",
        "readiness_profile_path",
    }
    converted = {
        name: (Path(item) if name in path_fields and item is not None else item)
        for name, item in value.items()
    }
    return HostLaunchConfig(**converted)


class LocalWorkerProcessPreparer:
    """Runtime-facing preparer that launches the pinned Worker over socketpair."""

    def __init__(
        self,
        config: HostLaunchConfig,
        *,
        environ: Mapping[str, str] | None = None,
        timeout_seconds: float = 900.0,
    ):
        self.config = config
        self.environ = dict(os.environ if environ is None else environ)
        self.timeout_seconds = float(timeout_seconds)
        self._active: PreparedWorkerHandle | None = None

    @staticmethod
    def _birth(pid: int) -> str:
        from source.runtime.worker.preflight import _process_birth_identity

        value = _process_birth_identity(pid)
        if not value:
            raise LauncherConfigurationError("prepared Worker birth identity is unavailable")
        return value

    def _rpc(self, handle: PreparedWorkerHandle, payload: Mapping[str, Any]) -> dict[str, Any]:
        if handle.closed or handle.worker.poll() is not None:
            raise LauncherConfigurationError("prepared Worker is not alive")
        if self._birth(handle.worker.pid) != handle.worker_birth_id:
            raise LauncherConfigurationError("prepared Worker identity changed")
        _send_private_frame(handle.control, payload)
        response = _receive_private_frame(handle.control)
        if response.get("version") != CONTROL_VERSION:
            raise LauncherConfigurationError("prepared Worker control version is invalid")
        if response.get("status") != "ok":
            raise LauncherConfigurationError(
                str(response.get("error") or "prepared Worker rejected the operation")
            )
        return response

    def prepare(self, profile: object, *, operation_id: str, channel_id: str) -> PreparedWorkerHandle:
        if self._active is not None:
            self.abort(self._active)
        parent, child = socket.socketpair()
        worker: subprocess.Popen[bytes] | None = None
        try:
            executable = Path(getattr(profile, "worker_executable"))
            worker_root = Path(__file__).resolve().parents[2]
            child_env = dict(self.environ)
            child_env["PYTHONPATH"] = str(worker_root)
            worker = subprocess.Popen(
                [
                    str(executable), "-m", "source.runtime.supervisor",
                    "--prepared-control-fd", str(child.fileno()),
                ],
                cwd=str(worker_root),
                env=child_env,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                pass_fds=(child.fileno(),),
            )
            child.close()
            parent.settimeout(self.timeout_seconds)
            handle = PreparedWorkerHandle(
                worker=worker,
                worker_birth_id=self._birth(worker.pid),
                control=parent,
                report_value={},
            )
            response = self._rpc(
                handle,
                {
                    "version": CONTROL_VERSION,
                    "command": "prepare",
                    "operation_id": operation_id,
                    "channel_id": channel_id,
                    "profile": _private_profile_payload(profile),
                    "config": _private_config_payload(self.config),
                },
            )
            report = response.get("report")
            if not isinstance(report, dict):
                raise LauncherConfigurationError("prepared Worker returned no process report")
            handle.report_value = report
            self._active = handle
            return handle
        except BaseException:
            child.close()
            parent.close()
            if worker is not None and worker.poll() is None:
                try:
                    # Closing the inherited control socket is the normal abort
                    # signal. Give the Worker time to clean its separately
                    # sessioned host and engine before terminating the Worker.
                    worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    _terminate_and_wait(worker, worker.pid)
            raise

    def report(self, handle: object) -> Mapping[str, Any]:
        if not isinstance(handle, PreparedWorkerHandle) or handle is not self._active:
            raise LauncherConfigurationError("prepared Worker handle is not active")
        response = self._rpc(
            handle, {"version": CONTROL_VERSION, "command": "report"}
        )
        report = response.get("report")
        if not isinstance(report, dict):
            raise LauncherConfigurationError("prepared Worker returned no process report")
        handle.report_value = report
        return report

    def activate(self, handle: object, grant: Mapping[str, Any]) -> None:
        if not isinstance(handle, PreparedWorkerHandle) or handle is not self._active:
            raise LauncherConfigurationError("prepared Worker handle is not active")
        self._rpc(
            handle,
            {"version": CONTROL_VERSION, "command": "activate", "grant": dict(grant)},
        )
        handle.activated = True

    def abort(self, handle: object) -> None:
        if not isinstance(handle, PreparedWorkerHandle) or handle is not self._active:
            return
        failure: BaseException | None = None
        try:
            self._rpc(handle, {"version": CONTROL_VERSION, "command": "abort"})
            handle.worker.wait(timeout=5)
        except BaseException as exc:
            failure = exc
        finally:
            handle.control.close()
            if handle.worker.poll() is None:
                try:
                    handle.worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    _terminate_and_wait(handle.worker, handle.worker.pid)
            handle.closed = True
            self._active = None
        if failure is not None:
            raise failure

    def reconnect(self, receipt: Mapping[str, Any]) -> object | None:
        handle = self._active
        if handle is None or handle.closed or not handle.activated:
            return None
        response = self._rpc(
            handle,
            {"version": CONTROL_VERSION, "command": "reconnect", "receipt": dict(receipt)},
        )
        return handle if response.get("reconnected") is True else None


def _serve_prepared_worker(descriptor: int) -> int:
    """Run the private Worker side of ``LocalWorkerProcessPreparer``."""

    control = socket.socket(fileno=descriptor)
    control.settimeout(None)
    adapter: LocalWorkerPreparerAdapter | None = None
    handle: PreparedHostHandle | None = None
    try:
        while True:
            try:
                request = _receive_private_frame(control)
                if request.get("version") != CONTROL_VERSION:
                    raise LauncherConfigurationError("private Worker control version is invalid")
                command = request.get("command")
                if command == "prepare":
                    if adapter is not None:
                        raise LauncherConfigurationError("private Worker is already prepared")
                    profile_value = request.get("profile")
                    config_value = request.get("config")
                    if not isinstance(profile_value, Mapping) or set(profile_value) != set(_PROFILE_FIELDS):
                        raise LauncherConfigurationError("private Worker profile has an invalid shape")
                    if not isinstance(config_value, Mapping):
                        raise LauncherConfigurationError("private Worker configuration is invalid")
                    profile = SimpleNamespace(
                        **{
                            name: Path(value)
                            if name.endswith("_root") or name.endswith("_executable")
                            else value
                            for name, value in profile_value.items()
                        }
                    )
                    adapter = LocalWorkerPreparerAdapter(
                        _config_from_private_payload(config_value), environ=os.environ
                    )
                    handle = adapter.prepare(
                        profile,
                        operation_id=str(request.get("operation_id") or ""),
                        channel_id=str(request.get("channel_id") or ""),
                    )
                    response: dict[str, Any] = {
                        "version": CONTROL_VERSION,
                        "status": "ok",
                        "report": dict(adapter.report(handle)),
                    }
                elif command == "report" and adapter is not None and handle is not None:
                    response = {
                        "version": CONTROL_VERSION,
                        "status": "ok",
                        "report": dict(adapter.report(handle)),
                    }
                elif command == "activate" and adapter is not None and handle is not None:
                    grant = request.get("grant")
                    if not isinstance(grant, Mapping):
                        raise LauncherConfigurationError("private Worker activation grant is invalid")
                    adapter.activate(handle, grant)
                    response = {"version": CONTROL_VERSION, "status": "ok"}
                elif command == "reconnect" and adapter is not None and handle is not None:
                    receipt = request.get("receipt")
                    if not isinstance(receipt, Mapping):
                        raise LauncherConfigurationError("private Worker reconnect receipt is invalid")
                    response = {
                        "version": CONTROL_VERSION,
                        "status": "ok",
                        "reconnected": adapter.reconnect(receipt) is handle,
                    }
                elif command == "abort" and adapter is not None and handle is not None:
                    adapter.abort(handle)
                    _send_private_frame(
                        control, {"version": CONTROL_VERSION, "status": "ok"}
                    )
                    return 0
                else:
                    raise LauncherConfigurationError("private Worker command is invalid")
                _send_private_frame(control, response)
            except LauncherConfigurationError as exc:
                _send_private_frame(
                    control,
                    {"version": CONTROL_VERSION, "status": "error", "error": str(exc)},
                )
    except (BrokenPipeError, ConnectionError, OSError):
        if adapter is not None and handle is not None:
            try:
                adapter.abort(handle)
            except BaseException:
                return 78
        return 78
    finally:
        control.close()


def launch_generic_pack_host(
    config: HostLaunchConfig,
    *,
    environ: Mapping[str, str] | None = None,
    ready_timeout_seconds: float = 20.0,
    enforce_readiness: bool | None = None,
) -> int:
    """Start exactly one host and return its exit status.

    The process is a fresh session leader, so signal forwarding and cleanup are
    limited to the group created for this launch. A host that never publishes a
    matching ready record is terminated and the Worker fails closed.
    """

    env = os.environ if environ is None else environ
    _reject_unissued_execution_target(env)
    if enforce_readiness is None:
        # Direct unit fixtures historically exercise process containment with a
        # non-Runtime endpoint.  Every supported loopback launch is gated.
        enforce_readiness = _loopback_endpoint(config.runtime_endpoint) or (config.support_root / "discovery.json").exists()
    owned_vibecomfy: _OwnedVibeComfySession | None = None
    if enforce_readiness:
        try:
            vibecomfy_session, owned_vibecomfy = _start_owned_vibecomfy_session(
                config, env
            )
            profile_path, profile_hash = _prepare_worker_readiness(
                config,
                env,
                vibecomfy_session=vibecomfy_session,
            )
            config = replace(config, readiness_profile_path=profile_path, readiness_profile_hash=profile_hash)
        except LauncherConfigurationError:
            if owned_vibecomfy is not None:
                _stop_owned_vibecomfy_session(owned_vibecomfy)
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if owned_vibecomfy is not None:
                _stop_owned_vibecomfy_session(owned_vibecomfy)
            (config.support_root / "worker-readiness-profile.json").unlink(missing_ok=True)
            raise LauncherConfigurationError("Worker readiness preparation failed") from exc
    argv = config.argv()
    child_env = _host_environment(env, config)
    config.ready_file.unlink(missing_ok=True)
    received: list[str] = []
    child: subprocess.Popen[bytes] | None = None

    def _forward_signal(signum: int, _frame: object) -> None:
        try:
            received.append(signal.Signals(signum).name)
        except ValueError:
            received.append(f"SIG{signum}")
        if child is not None:
            _signal_owned_group(child, signum)

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)
    owned_pgid: int | None = None
    cleanup_complete = False

    def _cleanup_host() -> None:
        nonlocal cleanup_complete, owned_vibecomfy
        if cleanup_complete:
            return
        cleanup_complete = True
        if child is not None and owned_pgid is not None:
            _terminate_and_wait(child, owned_pgid)
        if owned_vibecomfy is not None:
            _stop_owned_vibecomfy_session(owned_vibecomfy)
            owned_vibecomfy = None

    def _invalidate_profile() -> None:
        if config.readiness_profile_path is not None:
            config.readiness_profile_path.unlink(missing_ok=True)

    try:
        try:
            child = subprocess.Popen(
                argv,
                cwd=str(config.source_checkout),
                env=child_env,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            if os.name != "nt":
                try:
                    own_group = os.getpgid(child.pid) == child.pid
                except OSError:
                    own_group = False
                if not own_group:
                    child.terminate()
                    child.wait(timeout=3)
                    raise LauncherConfigurationError(
                        "GenericPackHost did not become its own process-group leader"
                    )
            owned_pgid = child.pid
        except BaseException:
            _invalidate_profile()
            _cleanup_host()
            raise

        try:
            state = {
                "status": "starting",
                "pid": child.pid,
                "pgid": owned_pgid,
                "argv": argv,
                "interpreter": str(config.host_python),
                "ready_file": str(config.ready_file),
                "state_file": str(config.state_file),
                "env_keys": sorted(child_env),
                "allowed_env": sorted(HOST_ENV_ALLOWLIST | {"PYTHONPATH"}),
            }
            _atomic_write_json(config.state_file, state)

            deadline = time.monotonic() + ready_timeout_seconds
            ready = False
            while time.monotonic() < deadline:
                if _ready_file_is_owned(config.ready_file, child):
                    ready = True
                    break
                if child.poll() is not None:
                    break
                time.sleep(0.05)

            if not ready:
                _cleanup_host()
                _invalidate_profile()
                returncode = _normalize_returncode(child.returncode)
                _atomic_write_json(
                    config.state_file,
                    {**state, "status": "failed", "returncode": returncode, "ready": False, "signals": received},
                )
                return returncode or 1

            _atomic_write_json(config.state_file, {**state, "status": "ready", "ready": True})
            returncode = _normalize_returncode(child.wait())
            _cleanup_host()
            _atomic_write_json(
                config.state_file,
                {
                    **state,
                    "status": "exited",
                    "ready": True,
                    "returncode": returncode,
                    "signals": received,
                },
            )
            return returncode
        except BaseException:
            _cleanup_host()
            _invalidate_profile()
            return 78
    finally:
        signal.signal(signal.SIGINT, previous_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous_handlers[signal.SIGTERM])


def main(argv: Sequence[str] | None = None) -> int:
    # Task/route arguments cannot select a host interpreter, source path,
    # engine, template, or backend.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        if len(arguments) == 2 and arguments[0] == "--prepared-control-fd":
            try:
                descriptor = int(arguments[1])
            except ValueError:
                print("Worker launcher configuration error: invalid private control fd", file=sys.stderr)
                return 78
            return _serve_prepared_worker(descriptor)
        print("Worker launcher configuration error: unsupported private arguments", file=sys.stderr)
        return 78
    try:
        config = HostLaunchConfig.from_environment()
        return launch_generic_pack_host(
            config,
            enforce_readiness=_loopback_endpoint(config.runtime_endpoint)
            or (config.support_root / "discovery.json").exists(),
        )
    except LauncherConfigurationError as exc:
        print(f"Worker launcher configuration error: {exc}", file=sys.stderr)
        return 78


__all__ = [
    "GENERIC_HOST_EXECUTOR_ID",
    "GENERIC_HOST_MODULE",
    "HOST_ENV_ALLOWLIST",
    "HostLaunchConfig",
    "LocalWorkerPreparerAdapter",
    "LocalWorkerProcessPreparer",
    "LauncherConfigurationError",
    "PreparedHostHandle",
    "PreparedWorkerHandle",
    "RuntimeDiscovery",
    "launch_generic_pack_host",
    "main",
]
