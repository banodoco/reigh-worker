"""Launch and supervise the one Astrid GenericPackHost owned by a Worker.

The Worker is deliberately not a task executor. It only validates the
operator-supplied host profile, starts one external GenericPackHost, publishes
neutral process state, forwards termination signals, and returns the host's
exit status unchanged.
"""

from __future__ import annotations

import json
import hashlib
import hmac
import math
import os
import ctypes
from pathlib import Path
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from source.runtime.custody_broker import CustodyError, RoleBoundCustodyBroker


GENERIC_HOST_MODULE = "astrid.core.execution.generic_host"
GENERIC_HOST_EXECUTOR_ID = "astrid-pack-host"
PREPARATION_VERSION = "runtime.local-worker-preparation/v2"
ACTIVATION_VERSION = "runtime.local-worker-activation/v1"
RECEIPT_VERSION = "runtime.local-worker-receipt/v3"
CONTROL_VERSION = "reigh.local-worker-control/v2"
ACTIVATION_ACCEPTED_VERSION = "astrid.local-worker-activation-accepted/v1"
HANDOFF_PAYLOAD_VERSION = "runtime.local-worker-handoff/v1"
HOST_CONTROL_VERSION = "astrid.local-worker-host-control/v1"
HANDOFF_RECORD_VERSION = "runtime.local-worker-handoff-record/v1"
HANDOFF_EXPORT_SEAL_VERSION = "runtime.local-worker-handoff-export-seal/v1"
# Handoff seal requests contain the full registered-state export.  Match the
# bounded Runtime-side private control ceiling so the exact sealed bytes can
# be verified without truncation or a second unbound transport.
_CONTROL_FRAME_LIMIT = 1024 * 1024
VIBECOMFY_CLEANUP_TOTAL_SECONDS = 35.0
VIBECOMFY_TERM_GRACE_SECONDS = 15.0

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


class _HandoffRejected(LauncherConfigurationError):
    """A handoff contender failed without authority to mutate custody."""


class _HandoffAuthorizedFailure(LauncherConfigurationError):
    """The bound handoff custodian failed and the owned graph must be cleaned."""

    def __init__(
        self,
        message: str,
        *,
        host_control_diagnostic: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.host_control_diagnostic = (
            dict(host_control_diagnostic)
            if host_control_diagnostic is not None
            else None
        )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LauncherConfigurationError("private control value is not canonical JSON") from exc


def _sha256_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


_PS_LSTART_BIRTH_ID = re.compile(
    r"ps-lstart:(Mon|Tue|Wed|Thu|Fri|Sat|Sun) +"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +"
    r"([1-9]|[12][0-9]|3[01]) +"
    r"([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9]) +([0-9]{4})\Z"
)


def _process_birth_identities_match(expected: object, observed: object) -> bool:
    if not isinstance(expected, str) or not isinstance(observed, str):
        return False
    if expected.startswith("ps-lstart:") or observed.startswith("ps-lstart:"):
        expected_match = _PS_LSTART_BIRTH_ID.fullmatch(expected)
        observed_match = _PS_LSTART_BIRTH_ID.fullmatch(observed)
        return (
            expected_match is not None
            and observed_match is not None
            and expected_match.groups() == observed_match.groups()
        )
    return bool(expected) and expected == observed


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        return False
    suffix = value[7:]
    return all(character in "0123456789abcdef" for character in suffix)


def _is_canonical_digest(value: object) -> bool:
    """Validate GenericPackHost's unprefixed canonical content digests."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonce_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LauncherConfigurationError("handoff deadline is invalid")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise LauncherConfigurationError("handoff deadline is invalid")
    return converted


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
    worker_credential_pending: bool
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
    source_checkout: Path | None
    pack_root: Path
    runtime_endpoint: str
    credential_file: Path
    support_root: Path
    runtime_instance_id: str
    ready_file: Path
    state_file: Path
    boot_manifest_path: Path
    boot_manifest_hash: str
    launch_mode: str = "editable"
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
        launch_mode = str(env.get("ASTRID_HOST_LAUNCH_MODE", "editable")).strip()
        if launch_mode not in {"editable", "installed"}:
            raise LauncherConfigurationError(
                "ASTRID_HOST_LAUNCH_MODE must be 'editable' or 'installed'"
            )
        source_checkout = (
            _resolved_path("ASTRID_HOST_SOURCE_CHECKOUT", env, directory=True)
            if launch_mode == "editable"
            else None
        )
        if launch_mode == "installed" and env.get("ASTRID_HOST_SOURCE_CHECKOUT", "").strip():
            raise LauncherConfigurationError(
                "installed host launch must not select ASTRID_HOST_SOURCE_CHECKOUT"
            )
        pack_root = _resolved_path("ASTRID_HOST_PACK_ROOT", env, directory=True)
        if source_checkout is not None and not pack_root.is_relative_to(source_checkout):
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
            launch_mode=launch_mode,
            capability_matrix=capability_matrix,
        )

    def argv(
        self,
        *,
        activation_fd: int | None = None,
        host_control_fd: int | None = None,
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
            "--runtime-instance-id",
            self.runtime_instance_id,
            "--register",
            "--boot-manifest-path",
            str(self.boot_manifest_path),
            "--boot-manifest-hash",
            self.boot_manifest_hash,
        ]
        if self.launch_mode == "editable":
            if self.source_checkout is None:
                raise LauncherConfigurationError(
                    "editable host launch requires a source checkout"
                )
            args.extend(("--source-checkout", str(self.source_checkout)))
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
        if host_control_fd is not None:
            if not isinstance(host_control_fd, int) or host_control_fd < 0:
                raise LauncherConfigurationError(
                    "GenericPackHost supervisor descriptor is invalid"
                )
            args.extend(("--host-control-fd", str(host_control_fd)))
        return args


def _host_environment(environ: Mapping[str, str], config: HostLaunchConfig) -> dict[str, str]:
    child_env = {key: value for key, value in environ.items() if key in HOST_ENV_ALLOWLIST}
    # Ambient PYTHONPATH is never inherited. Editable mode admits exactly its
    # configured checkout; installed mode relies only on the host interpreter.
    child_env.pop("PYTHONPATH", None)
    if config.launch_mode == "editable":
        if config.source_checkout is None:
            raise LauncherConfigurationError(
                "editable host launch requires a source checkout"
            )
        child_env["PYTHONPATH"] = str(config.source_checkout)
    child_env["PYTHONUNBUFFERED"] = "1"
    if config.readiness_profile_path is not None and config.readiness_profile_hash is not None:
        child_env["ASTRID_HOST_READINESS_PROFILE_PATH"] = str(config.readiness_profile_path)
        child_env["ASTRID_HOST_READINESS_PROFILE_HASH"] = config.readiness_profile_hash
    return child_env


def _host_working_directory(config: HostLaunchConfig) -> Path:
    if config.launch_mode == "installed":
        return config.support_root
    if config.launch_mode != "editable" or config.source_checkout is None:
        raise LauncherConfigurationError("host launch mode is invalid")
    return config.source_checkout


def _host_artifact_root(config: HostLaunchConfig) -> Path:
    """Return the verified code root used only for local readiness facts."""
    if config.launch_mode == "installed":
        return config.pack_root.parent
    return _host_working_directory(config)


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


@dataclass(frozen=True)
class _CleanupIdentity:
    pid: int
    birth_id: str
    uid: int
    parent_pid: int
    process_group: int
    session_id: int
    executable: Path
    artifact_digest: str
    argv_digest: str


def _cleanup_ps(pid: int, field: str) -> str:
    result = subprocess.run(
        ["ps", "-p", str(int(pid)), "-o", f"{field}="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise LauncherConfigurationError("owned process cleanup identity is unavailable")
    return result.stdout.strip().splitlines()[0].strip()


def _cleanup_executable(pid: int) -> Path:
    try:
        if sys.platform == "darwin":
            library = ctypes.CDLL("/usr/lib/libproc.dylib")
            buffer = ctypes.create_string_buffer(4096)
            if library.proc_pidpath(int(pid), buffer, len(buffer)) <= 0:
                raise OSError("proc_pidpath failed")
            return Path(buffer.value.decode()).resolve()
        return Path(os.readlink(f"/proc/{int(pid)}/exe")).resolve()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LauncherConfigurationError(
            "owned process cleanup executable is unavailable"
        ) from exc


def _cleanup_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LauncherConfigurationError(
            "owned process cleanup executable cannot be read"
        ) from exc
    return "sha256:" + digest.hexdigest()


def _darwin_cleanup_argv(pid: int) -> tuple[bytes, ...] | None:
    """Read exact argv bytes through Darwin's supported KERN_PROCARGS2 API."""

    ctl_kern = 1
    kern_procargs2 = 49
    libc = ctypes.CDLL(None, use_errno=True)
    sysctl = libc.sysctl
    sysctl.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    sysctl.restype = ctypes.c_int
    mib = (ctypes.c_int * 3)(ctl_kern, kern_procargs2, int(pid))
    size = ctypes.c_size_t(0)
    if sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value < 4:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        return None
    raw = buffer.raw[: size.value]
    argc = struct.unpack_from("=i", raw)[0]
    if argc < 1 or argc > 1_000_000:
        return None
    offset = 4
    executable_end = raw.find(b"\0", offset)
    if executable_end < 0:
        return None
    offset = executable_end + 1
    while offset < len(raw) and raw[offset] == 0:
        offset += 1
    argv: list[bytes] = []
    while len(argv) < argc and offset < len(raw):
        end = raw.find(b"\0", offset)
        if end < 0:
            return None
        argv.append(raw[offset:end])
        offset = end + 1
    return tuple(argv) if len(argv) == argc else None


def _cleanup_argv(pid: int) -> tuple[bytes, ...]:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        raw = b""
    if raw:
        argv = tuple(value for value in raw.split(b"\0") if value)
        if argv:
            return argv
    if sys.platform == "darwin":
        argv = _darwin_cleanup_argv(pid)
        if argv:
            return argv
    raise LauncherConfigurationError("owned process cleanup argv is unavailable")


def _argv_digest(argv: Sequence[bytes]) -> str:
    encoded = bytearray(b"astrid.argv.v1\0")
    encoded.extend(len(argv).to_bytes(8, "big"))
    for value in argv:
        if not isinstance(value, bytes):
            raise LauncherConfigurationError("owned process cleanup argv is invalid")
        encoded.extend(len(value).to_bytes(8, "big"))
        encoded.extend(value)
    return "sha256:" + hashlib.sha256(bytes(encoded)).hexdigest()


def _capture_cleanup_identity(pid: int) -> _CleanupIdentity:
    from source.runtime.worker.preflight import _process_birth_identity

    birth = _process_birth_identity(pid)
    if not birth:
        raise LauncherConfigurationError("owned process cleanup birth identity is unavailable")
    executable = _cleanup_executable(pid)
    try:
        return _CleanupIdentity(
            pid=int(pid),
            birth_id=birth,
            uid=int(_cleanup_ps(pid, "uid")),
            parent_pid=int(_cleanup_ps(pid, "ppid")),
            process_group=os.getpgid(pid),
            session_id=os.getsid(pid),
            executable=executable,
            artifact_digest=_cleanup_digest(executable),
            argv_digest=_argv_digest(_cleanup_argv(pid)),
        )
    except (OSError, ValueError) as exc:
        raise LauncherConfigurationError("owned process cleanup identity is invalid") from exc


def _custody_identity(pid: int) -> Mapping[str, object] | None:
    try:
        identity = _capture_cleanup_identity(pid)
    except LauncherConfigurationError:
        return None
    return {
        "pid": identity.pid,
        "birth_id": identity.birth_id,
        "uid": identity.uid,
        "parent_pid": identity.parent_pid,
        "process_group": identity.process_group,
        "session_id": identity.session_id,
        "executable": str(identity.executable),
        "artifact_digest": identity.artifact_digest,
        "argv_digest": identity.argv_digest,
    }


@dataclass
class _CustodyLaunchOwner:
    """Caller-visible ownership populated before custody admission can fail."""

    role: str
    process: subprocess.Popen[bytes] | None = None
    broker: RoleBoundCustodyBroker | None = None
    state: str = "new"
    error: BaseException | None = None


class _CustodyAdmissionFailure(LauncherConfigurationError):
    def __init__(self, owner: _CustodyLaunchOwner):
        self.owner = owner
        suffix = (
            " and live custody remains unresolved"
            if owner.state == "unresolved"
            else " after audit-token cleanup and reaping"
        )
        super().__init__(f"{owner.role} audit-token custody registration failed{suffix}")


_CUSTODY_LAUNCH_LOCK = threading.Lock()
_UNRESOLVED_CUSTODY: dict[int, _CustodyLaunchOwner] = {}


def _reserve_custody_launch(owner: _CustodyLaunchOwner) -> None:
    with _CUSTODY_LAUNCH_LOCK:
        unresolved = [item for item in _UNRESOLVED_CUSTODY.values() if item.state == "unresolved"]
        if unresolved:
            roles = ", ".join(sorted({item.role for item in unresolved}))
            raise LauncherConfigurationError(
                f"unresolved audit-token custody refuses follow-on launch ({roles})"
            )
        if owner.state != "new" or owner.process is not None or owner.broker is not None:
            raise LauncherConfigurationError("custody launch owner is already used")
        owner.state = "reserved"
        _UNRESOLVED_CUSTODY[id(owner)] = owner


def _release_custody_launch(owner: _CustodyLaunchOwner, state: str) -> None:
    owner.state = state
    with _CUSTODY_LAUNCH_LOCK:
        _UNRESOLVED_CUSTODY.pop(id(owner), None)


def _cleanup_failed_custody(owner: _CustodyLaunchOwner) -> bool:
    """Attempt bounded audit-token-only cleanup; retain uncertainty on failure."""

    process = owner.process
    broker = owner.broker
    if process is None or broker is None:
        return True
    if process.poll() is not None:
        try:
            process.wait(timeout=0)
        except (OSError, subprocess.SubprocessError):
            return False
        return True
    try:
        broker.signal_failed_admission(signal.SIGTERM, expected_pid=process.pid)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            broker.signal_failed_admission(signal.SIGKILL, expected_pid=process.pid)
            process.wait(timeout=3)
    except (CustodyError, OSError, subprocess.SubprocessError):
        return False
    return process.poll() is not None


def _custodied_popen(
    argv: Sequence[str], *, custody_role: str,
    custody_owner: _CustodyLaunchOwner | None = None, **kwargs: Any
) -> subprocess.Popen[bytes]:
    """Spawn through the reviewed pre-exec audit-token broker contract."""

    owner = custody_owner or _CustodyLaunchOwner(custody_role)
    if owner.role != custody_role:
        raise LauncherConfigurationError("custody launch owner role is invalid")
    _reserve_custody_launch(owner)
    options = dict(kwargs)
    start_new_session = bool(options.pop("start_new_session", False))
    child_environment = dict(options.pop("env", os.environ))
    try:
        broker = RoleBoundCustodyBroker(
            role=custody_role,
            identity_provider=_custody_identity,
        )
        child_environment.update(
            broker.child_environment(argv, start_new_session=start_new_session)
        )
    except CustodyError as exc:
        owner.error = exc
        _release_custody_launch(owner, "failed_before_spawn")
        raise LauncherConfigurationError(
            f"{custody_role} audit-token custody admission is unavailable"
        ) from exc
    options["env"] = child_environment
    options["start_new_session"] = False
    wrapper = [
        sys.executable,
        "-I",
        str(Path(__file__).with_name("custody_broker.py").resolve()),
        "--custody-exec",
    ]
    try:
        process = subprocess.Popen(wrapper, **options)
    except BaseException:
        _release_custody_launch(owner, "failed_before_spawn")
        raise
    process._reigh_custody_broker = broker  # type: ignore[attr-defined]
    owner.process = process
    owner.broker = broker
    owner.state = "admission_pending"
    try:
        broker.wait_until_sealed()
    except CustodyError as exc:
        owner.error = exc
        owner.state = "unresolved"
        if _cleanup_failed_custody(owner):
            _release_custody_launch(owner, "failed_reaped")
        raise _CustodyAdmissionFailure(owner) from exc
    _release_custody_launch(owner, "sealed")
    return process


def _verify_cleanup_identity(
    identity: _CleanupIdentity, *, allow_reparented: bool = False
) -> bool:
    from source.runtime.worker.preflight import _process_birth_identity

    observed = _process_birth_identity(identity.pid)
    if observed is None:
        return False
    if observed != identity.birth_id:
        raise LauncherConfigurationError("owned process cleanup birth identity changed")
    try:
        current = _capture_cleanup_identity(identity.pid)
    except LauncherConfigurationError:
        if _process_birth_identity(identity.pid) is None:
            return False
        # On Darwin proc_pidpath stops exposing the executable once an owned
        # child has exited, while ps may still expose its unreaped zombie and
        # stable birth identity.  A zombie cannot own a listener or receive a
        # signal; classify it as absent here so the retained Popen can be
        # reaped below.  Any live or unobservable incarnation still fails
        # closed.
        try:
            state = _cleanup_ps(identity.pid, "state")
        except LauncherConfigurationError:
            if _process_birth_identity(identity.pid) is None:
                return False
            raise
        if state.startswith("Z"):
            return False
        raise
    comparable = current
    if allow_reparented and current.parent_pid == 1:
        comparable = replace(current, parent_pid=identity.parent_pid)
    if comparable != identity:
        raise LauncherConfigurationError("owned process cleanup identity changed")
    return True


def _signal_owned_group(process: subprocess.Popen[bytes], signum: int) -> None:
    if process.poll() is not None:
        return
    identity = _capture_cleanup_identity(process.pid)
    if identity.process_group != identity.pid or identity.session_id != identity.pid:
        raise LauncherConfigurationError("owned process cleanup group is invalid")
    broker = getattr(process, "_reigh_custody_broker", None)
    if not isinstance(broker, RoleBoundCustodyBroker):
        raise LauncherConfigurationError(
            "owned process role-bound audit-token custody is unavailable"
        )
    if _verify_cleanup_identity(identity):
        try:
            broker.signal(signum, expected_pid=identity.pid)
        except CustodyError as exc:
            raise LauncherConfigurationError(
                "owned process audit-token signal failed"
            ) from exc


def _terminate_and_wait(process: subprocess.Popen[bytes], pgid: int) -> None:
    """Terminate and reap a verified host process group."""

    identity = _capture_cleanup_identity(process.pid)
    if (
        pgid != identity.pid
        or identity.process_group != identity.pid
        or identity.session_id != identity.pid
    ):
        raise LauncherConfigurationError("owned process cleanup group is invalid")
    _signal_owned_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if _verify_cleanup_identity(identity):
            _signal_owned_group(process, signal.SIGKILL)
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
    daemon_fields = {
        "version", "endpoint", "pid", "process_birth_id", "active_realm", "runtime_instance_id",
        "realm_root", "protocol_version", "schema_version", "coordinator_epoch", "credential_file",
        "worker_credential_file", "worker_credential_pending", "worker_actor", "worker_scopes",
    }
    operator_fields = daemon_fields | {"capability_digest", "advertised_at"}
    observed_fields = frozenset(record) if isinstance(record, dict) else frozenset()
    if observed_fields not in {frozenset(daemon_fields), frozenset(operator_fields)}:
        raise LauncherConfigurationError("Runtime discovery.json schema is invalid")
    if observed_fields == operator_fields:
        capability_digest = record.get("capability_digest")
        advertised_at = record.get("advertised_at")
        if (
            not isinstance(capability_digest, str)
            or (
                capability_digest != ""
                and (
                    len(capability_digest) != 71
                    or not capability_digest.startswith("sha256:")
                    or any(character not in "0123456789abcdef" for character in capability_digest[7:])
                )
            )
            or isinstance(advertised_at, bool)
            or not isinstance(advertised_at, (int, float))
            or not math.isfinite(advertised_at)
            or advertised_at <= 0
        ):
            raise LauncherConfigurationError("Runtime operator discovery metadata is invalid")
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
    credential_pending = record.get("worker_credential_pending")
    if not isinstance(credential_pending, bool):
        raise LauncherConfigurationError("Runtime worker credential pending state is invalid")
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
        elif not credential_pending:
            raise LauncherConfigurationError(
                "Runtime worker credential is missing without a pending handoff"
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
        worker_credential_file=worker_path, worker_credential_pending=credential_pending,
        worker_actor=worker_actor, worker_scopes=tuple(worker_scopes),
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
        repo_root=_host_artifact_root(config),
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
            "host_interpreter": str(config.host_python),
            "launch_mode": config.launch_mode,
            "source_checkout": str(config.source_checkout) if config.source_checkout is not None else None,
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
        from source.runtime.worker.preflight import _process_birth_identity
        if _process_birth_identity(pid) is None:
            raise ValueError("session daemon process identity is unavailable")
        comfy_pid = int(registry["comfy_pid"].read_text(encoding="utf-8").strip())
        if comfy_pid <= 0:
            raise ValueError("comfy_pid must be positive")
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
    daemon_cleanup_identity: _CleanupIdentity | None = None
    listener_cleanup_identity: _CleanupIdentity | None = None

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
    custody_owner = _CustodyLaunchOwner("engine_daemon")
    try:
        log_handle = log_path.open("ab", buffering=0)
        process = _custodied_popen(
            command,
            custody_role="engine_daemon",
            custody_owner=custody_owner,
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
                    daemon_cleanup_identity=_capture_cleanup_identity(process.pid),
                    listener_cleanup_identity=_capture_cleanup_identity(
                        int(payload["comfy_pid"])
                    ),
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
                daemon_cleanup_identity=_capture_cleanup_identity(process.pid),
                listener_cleanup_identity=_capture_cleanup_identity(
                    int(partial["comfy_pid"])
                ),
            )
        _stop_owned_vibecomfy_session(cleanup_session)
        raise


def _stop_owned_vibecomfy_session(session: _OwnedVibeComfySession) -> None:
    process = session.process
    parsed = urlsplit(session.server_url)
    daemon = session.daemon_cleanup_identity
    listener = session.listener_cleanup_identity
    if daemon is None or listener is None:
        if process.poll() is None or _owned_listener_pid(parsed.port or 0) is not None:
            raise LauncherConfigurationError(
                "owned VibeComfy cleanup identity is incomplete"
            )
        return
    if daemon.pid != process.pid or listener.pid != session.comfy_pid:
        raise LauncherConfigurationError("owned VibeComfy cleanup identity is inconsistent")
    if (
        daemon.process_group != daemon.pid
        or daemon.session_id != daemon.pid
        or listener.process_group != daemon.pid
        or listener.session_id != daemon.pid
    ):
        raise LauncherConfigurationError("owned VibeComfy cleanup group is invalid")

    def live_members() -> list[_CleanupIdentity]:
        result: list[_CleanupIdentity] = []
        if _verify_cleanup_identity(daemon, allow_reparented=False):
            result.append(daemon)
        if _verify_cleanup_identity(listener, allow_reparented=True):
            result.append(listener)
        owner = _owned_listener_pid(parsed.port or 0)
        if listener in result:
            if owner != listener.pid:
                raise LauncherConfigurationError(
                    "owned VibeComfy listener identity changed"
                )
        elif owner is not None:
            raise LauncherConfigurationError(
                "owned VibeComfy listener was replaced"
            )
        return result

    cleanup_deadline = time.monotonic() + VIBECOMFY_CLEANUP_TOTAL_SECONDS
    members = live_members()
    if members:
        # Full identity, group and endpoint ownership are rechecked directly
        # before both TERM and KILL.  No PID-only signal is used.
        _signal_owned_group(process, signal.SIGTERM)
        term_deadline = min(
            cleanup_deadline,
            time.monotonic() + VIBECOMFY_TERM_GRACE_SECONDS,
        )
        while time.monotonic() < term_deadline:
            if not any(
                _verify_cleanup_identity(item, allow_reparented=item is listener)
                for item in (daemon, listener)
            ):
                break
            time.sleep(0.1)
        members = live_members()
        if members:
            _signal_owned_group(process, signal.SIGKILL)
            while time.monotonic() < cleanup_deadline:
                if not any(
                    _verify_cleanup_identity(item, allow_reparented=item is listener)
                    for item in (daemon, listener)
                ):
                    break
                time.sleep(0.1)
    if live_members():
        raise LauncherConfigurationError("owned VibeComfy group survived cleanup")
    if process.poll() is None:
        try:
            remaining = cleanup_deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("owned VibeComfy daemon", 0)
            process.wait(timeout=min(1.0, remaining))
        except subprocess.TimeoutExpired as exc:
            raise LauncherConfigurationError(
                "owned VibeComfy daemon could not be reaped"
            ) from exc


@dataclass
class PreparedHostHandle:
    """Worker-owned engine, parked host, and inherited activation endpoint."""

    profile: object
    operation_id: str
    channel_id: str
    host: subprocess.Popen[bytes]
    host_birth_id: str
    host_cleanup_identity: _CleanupIdentity
    activation: socket.socket
    host_control: socket.socket
    engine: _OwnedVibeComfySession
    engine_report: dict[str, object]
    readiness_profile: Path | None = None
    activated: bool = False
    activation_grant: dict[str, object] | None = None
    closed: bool = False


@dataclass
class _WorkerHandoff:
    handoff_id: str
    nonce_digest: str
    sealed_record_digest: str
    deadline_monotonic: float
    deadline_unix_ms: int
    old_runtime: dict[str, Any]
    receipt_evidence_digest: str
    credential_generation: dict[str, Any]
    registered_state: dict[str, Any]
    old_owner: dict[str, Any]
    sealed_record: dict[str, Any]
    export_sealed_digest: str | None = None
    export_record_digest: str | None = None
    export_digest: str | None = None
    adopter_record_digest: str | None = None
    new_owner: dict[str, Any] | None = None
    phase: str = "paused"
    adopter_request_id: str | None = None
    new_runtime: dict[str, Any] | None = None
    nonce_consumed: bool = False
    acknowledgements: dict[str, tuple[str, dict[str, Any]]] | None = None


@dataclass(frozen=True)
class _FinalizedHandoffAck:
    handoff_id: str
    request_digest: str
    response: dict[str, Any]


_RUNTIME_IDENTITY_FIELDS = frozenset(
    {
        "endpoint",
        "protocol",
        "schema_digest",
        "runtime_epoch",
        "runtime_instance_id",
        "runtime_session_id",
    }
)
_CREDENTIAL_GENERATION_FIELDS = frozenset(
    {"generation", "token_sha256", "metadata_sha256", "commit_sha256"}
)
_REGISTERED_STATE_FIELDS = frozenset(
    {
        "executor_id", "source_epoch", "runtime", "capabilities",
        "registration_actor", "registration_bodies", "registration_allowlist",
    }
)
_CAPABILITY_STATE_FIELDS = frozenset(
    {
        "capability_id",
        "capability_digest",
        "source_digest",
        "dependency_digest",
        "ready",
        "preflight_digest",
    }
)
_OWNER_FIELDS = frozenset({"pid", "birth_id"})
_SEALED_OWNER_FIELDS = frozenset(
    {"pid", "birth_id", "runtime_instance_id", "runtime"}
)


def _strict_object(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise LauncherConfigurationError(f"{label} has an invalid shape")
    return dict(value)


def _runtime_identity(value: object, label: str) -> dict[str, Any]:
    result = _strict_object(value, _RUNTIME_IDENTITY_FIELDS, label)
    for name in _RUNTIME_IDENTITY_FIELDS - {"runtime_epoch"}:
        if not isinstance(result[name], str) or not result[name]:
            raise LauncherConfigurationError(f"{label} has an invalid value")
    if (
        isinstance(result["runtime_epoch"], bool)
        or not isinstance(result["runtime_epoch"], int)
        or result["runtime_epoch"] < 1
    ):
        raise LauncherConfigurationError(f"{label} Runtime epoch is invalid")
    if not _is_sha256(result["schema_digest"]):
        raise LauncherConfigurationError(f"{label} schema digest is invalid")
    return result


def _runtime_owner(value: object, label: str) -> dict[str, Any]:
    result = _strict_object(value, _OWNER_FIELDS, label)
    pid = result["pid"]
    birth_id = result["birth_id"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise LauncherConfigurationError(f"{label} pid is invalid")
    if not isinstance(birth_id, str) or not birth_id or len(birth_id) > 512:
        raise LauncherConfigurationError(f"{label} birth identity is invalid")
    return result


def _contains_raw_nonce(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in {"nonce", "raw_nonce"} or _contains_raw_nonce(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_nonce(item) for item in value)
    return False


_BOUNDED_HANDOFF_ERROR_CODES = {
    "private Worker handoff request has an invalid shape": "request_shape",
    "private Worker control version is invalid": "control_version",
    "private Worker handoff identity is invalid": "handoff_identity",
    "private Worker handoff digest is invalid": "handoff_digest",
    "private Worker handoff is already prepared": "already_prepared",
    "private Worker handoff deadline expired": "deadline_expired",
    "private Worker owner A is stale": "owner_a_stale",
    "private Worker sealed handoff owner is invalid": "sealed_owner_invalid",
    "private Worker sealed handoff owner does not match owner A": "sealed_owner_mismatch",
    "private Worker sealed handoff record is invalid": "sealed_record_invalid",
    "private Worker sealed handoff digest is invalid": "sealed_digest_invalid",
    "private Worker handoff record digest is invalid": "record_digest_invalid",
    "handoff credential generation changed": "credential_generation_changed",
    "handoff receipt evidence digest is invalid": "receipt_digest_invalid",
    "handoff receipt evidence does not match activation": "receipt_activation_mismatch",
    "GenericPackHost old Runtime state changed": "host_runtime_changed",
    "GenericPackHost control channel failed": "host_control_failed",
    "GenericPackHost control channel closed": "host_control_closed",
    "GenericPackHost control channel sent an unsolicited frame": "host_unsolicited_frame",
    "GenericPackHost acknowledgement binding is invalid": "host_ack_binding",
    "GenericPackHost acknowledgement phase is invalid": "host_ack_phase",
    "GenericPackHost acknowledgement digest is invalid": "host_ack_digest",
    "GenericPackHost activation identity changed": "host_activation_changed",
}


def _bounded_handoff_error_code(exc: BaseException) -> str:
    return _BOUNDED_HANDOFF_ERROR_CODES.get(
        str(exc),
        "authorized_failure"
        if isinstance(exc, _HandoffAuthorizedFailure)
        else "handoff_rejected"
        if isinstance(exc, _HandoffRejected)
        else "worker_configuration",
    )


def _authorized_handoff_error_response(
    exc: _HandoffAuthorizedFailure,
    command: object,
) -> dict[str, Any]:
    error_code = _bounded_handoff_error_code(exc)
    response: dict[str, Any] = {
        "version": CONTROL_VERSION,
        "status": "error",
        "error": "prepared Worker rejected the handoff",
        "error_code": error_code,
        "error_stage": (
            "control"
            if error_code.startswith("host_")
            else str(command or "control")
        ),
    }
    if exc.host_control_diagnostic is not None:
        response["host_control_diagnostic"] = dict(exc.host_control_diagnostic)
    return response


def _sealed_handoff_record(
    value: object,
    *,
    handoff_id: str,
    nonce_digest: str,
    sealed_record_digest: str,
    old_owner: Mapping[str, Any],
    old_runtime: Mapping[str, Any],
    deadline_monotonic: float,
    deadline_unix_ms: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _HandoffRejected("private Worker sealed handoff record is invalid")
    result = dict(value)
    record_digest = result.get("record_digest")
    try:
        sealed_owner = _strict_object(
            result.get("old_owner"),
            _SEALED_OWNER_FIELDS,
            "private Worker sealed handoff owner",
        )
        sealed_owner_projection = _runtime_owner(
            {
                "pid": sealed_owner["pid"],
                "birth_id": sealed_owner["birth_id"],
            },
            "private Worker sealed handoff owner",
        )
        sealed_runtime = _runtime_identity(
            sealed_owner["runtime"],
            "private Worker sealed handoff Runtime",
        )
    except LauncherConfigurationError as exc:
        raise _HandoffRejected(
            "private Worker sealed handoff owner is invalid"
        ) from exc
    if (
        sealed_owner_projection != dict(old_owner)
        or sealed_runtime != dict(old_runtime)
        or sealed_owner["runtime_instance_id"]
        != old_runtime["runtime_instance_id"]
    ):
        raise _HandoffRejected(
            "private Worker sealed handoff owner does not match owner A"
        )
    if (
        result.get("version") != HANDOFF_RECORD_VERSION
        or result.get("state") != "OWNED"
        or result.get("handoff_id") != handoff_id
        or result.get("nonce_digest") != nonce_digest
        or result.get("sealed_record_digest") != sealed_record_digest
        or result.get("deadline_monotonic") != deadline_monotonic
        or result.get("deadline_unix_ms") != deadline_unix_ms
        or not _is_sha256(record_digest)
        or _contains_raw_nonce(result)
    ):
        raise _HandoffRejected("private Worker sealed handoff record is invalid")
    seal_value = {
        key: item
        for key, item in result.items()
        if key not in {"sealed_record_digest", "record_digest"}
    }
    if not hmac.compare_digest(sealed_record_digest, _sha256_json(seal_value)):
        raise _HandoffRejected("private Worker sealed handoff digest is invalid")
    record_value = {
        key: item for key, item in result.items() if key != "record_digest"
    }
    if not hmac.compare_digest(str(record_digest), _sha256_json(record_value)):
        raise _HandoffRejected("private Worker handoff record digest is invalid")
    return result


def _credential_generation(value: object) -> dict[str, Any]:
    result = _strict_object(
        value, _CREDENTIAL_GENERATION_FIELDS, "handoff credential generation"
    )
    if not isinstance(result["generation"], str) or not result["generation"]:
        raise LauncherConfigurationError("handoff credential generation is invalid")
    for name in ("token_sha256", "metadata_sha256", "commit_sha256"):
        if not _is_sha256(result[name]):
            raise LauncherConfigurationError(f"handoff credential {name} is invalid")
    return result


def _registered_state(value: object) -> dict[str, Any]:
    result = _strict_object(value, _REGISTERED_STATE_FIELDS, "registered state")
    if not isinstance(result["executor_id"], str) or not result["executor_id"]:
        raise LauncherConfigurationError("registered executor identity is invalid")
    if not isinstance(result["source_epoch"], str) or not result["source_epoch"]:
        raise LauncherConfigurationError("registered source epoch is invalid")
    result["runtime"] = _runtime_identity(result["runtime"], "registered Runtime")
    capabilities = result["capabilities"]
    if not isinstance(capabilities, list):
        raise LauncherConfigurationError("registered capabilities are invalid")
    normalized: list[dict[str, Any]] = []
    for capability in capabilities:
        item = _strict_object(
            capability, _CAPABILITY_STATE_FIELDS, "registered capability"
        )
        if not isinstance(item["capability_id"], str) or not item["capability_id"]:
            raise LauncherConfigurationError("registered capability identity is invalid")
        if not isinstance(item["ready"], bool):
            raise LauncherConfigurationError("registered capability readiness is invalid")
        for name in ("capability_digest", "preflight_digest"):
            if not _is_sha256(item[name]):
                raise LauncherConfigurationError(
                    f"registered capability {name} is invalid"
                )
        for name in ("source_digest", "dependency_digest"):
            if not _is_canonical_digest(item[name]):
                raise LauncherConfigurationError(
                    f"registered capability {name} is invalid"
                )
        normalized.append(item)
    if [item["capability_id"] for item in normalized] != sorted(
        item["capability_id"] for item in normalized
    ):
        raise LauncherConfigurationError("registered capabilities are not canonical")
    result["capabilities"] = normalized
    actor = result["registration_actor"]
    bodies = result["registration_bodies"]
    allowlist = result["registration_allowlist"]
    routes = ("/v1/capabilities", "/v1/executors")
    if actor != result["executor_id"]:
        raise LauncherConfigurationError("registered admission actor is invalid")
    if not isinstance(bodies, Mapping) or set(bodies) != set(routes):
        raise LauncherConfigurationError("registered admission routes are invalid")
    normalized_bodies: dict[str, list[object]] = {}
    expected_allowlist = []
    for path in routes:
        items = bodies[path]
        if not isinstance(items, list):
            raise LauncherConfigurationError("registered admission bodies are invalid")
        normalized_bodies[path] = list(items)
        expected_allowlist.append(
            {
                "method": "POST",
                "path": path,
                "actor": actor,
                "body_sha256": sorted(_sha256_json(item) for item in items),
            }
        )
    expected_allowlist.sort(key=lambda item: (item["path"], item["actor"]))
    if allowlist != expected_allowlist:
        raise LauncherConfigurationError("registered admission allowlist is invalid")
    result["registration_bodies"] = normalized_bodies
    result["registration_allowlist"] = expected_allowlist
    return result


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
        self._handoff: _WorkerHandoff | None = None
        self._runtime_owner_identity: dict[str, Any] | None = None
        self._finalized_handoff_acks: list[_FinalizedHandoffAck] = []

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
        parent_activation, host_activation = socket.socketpair()
        parent_supervisor, host_supervisor = socket.socketpair()
        self.config.ready_file.unlink(missing_ok=True)
        child: subprocess.Popen[bytes] | None = None
        child_cleanup_identity: _CleanupIdentity | None = None
        custody_owner = _CustodyLaunchOwner("generic_pack_host")
        try:
            argv = self.config.argv(
                activation_fd=host_activation.fileno(),
                host_control_fd=host_supervisor.fileno(),
                operation_id=operation_id,
                channel_id=channel_id,
                activation_timeout_seconds=self.activation_timeout_seconds,
            )
            child = _custodied_popen(
                argv,
                custody_role="generic_pack_host",
                custody_owner=custody_owner,
                cwd=str(_host_working_directory(self.config)),
                env=_host_environment(self.environ, self.config),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                pass_fds=(host_activation.fileno(), host_supervisor.fileno()),
            )
            host_activation.close()
            host_supervisor.close()
            birth = self._birth(child.pid)
            child_cleanup_identity = _capture_cleanup_identity(child.pid)
            handle = PreparedHostHandle(
                profile=profile,
                operation_id=operation_id,
                channel_id=channel_id,
                host=child,
                host_birth_id=birth,
                host_cleanup_identity=child_cleanup_identity,
                activation=parent_activation,
                host_control=parent_supervisor,
                engine=engine,
                engine_report=dict(engine_report),
            )
            self._assert_live(handle, parked=True)
            self._active = handle
            return handle
        except BaseException:
            host_activation.close()
            host_supervisor.close()
            parent_activation.close()
            parent_supervisor.close()
            if child is not None:
                try:
                    if child_cleanup_identity is None:
                        child_cleanup_identity = _capture_cleanup_identity(child.pid)
                    if _verify_cleanup_identity(child_cleanup_identity):
                        _signal_owned_group(child, signal.SIGTERM)
                        try:
                            child.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            if _verify_cleanup_identity(child_cleanup_identity):
                                _signal_owned_group(child, signal.SIGKILL)
                            child.wait(timeout=3)
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

    @staticmethod
    def _ack_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result["ack_sha256"] = _sha256_json(result)
        return result

    @staticmethod
    def _common_handoff_request(
        request: Mapping[str, Any], expected: frozenset[str]
    ) -> tuple[str, str, str, float, int]:
        common = frozenset(
            {
                "version", "command", "handoff_id", "nonce_digest",
                "sealed_record_digest", "deadline_monotonic", "deadline_unix_ms",
            }
        )
        if set(request) != common | expected:
            raise _HandoffRejected("private Worker handoff request has an invalid shape")
        if request.get("version") != CONTROL_VERSION:
            raise _HandoffRejected("private Worker control version is invalid")
        handoff_id = request.get("handoff_id")
        nonce_digest = request.get("nonce_digest")
        sealed_digest = request.get("sealed_record_digest")
        if not isinstance(handoff_id, str) or not handoff_id or len(handoff_id) > 256:
            raise _HandoffRejected("private Worker handoff identity is invalid")
        if not _is_sha256(nonce_digest) or not _is_sha256(sealed_digest):
            raise _HandoffRejected("private Worker handoff digest is invalid")
        try:
            deadline_monotonic = _finite_number(request.get("deadline_monotonic"))
            unix_value = request.get("deadline_unix_ms")
            if isinstance(unix_value, bool) or not isinstance(unix_value, int) or unix_value < 1:
                raise LauncherConfigurationError("handoff deadline is invalid")
            deadline_unix_ms = unix_value
        except LauncherConfigurationError as exc:
            raise _HandoffRejected(str(exc)) from exc
        return (
            handoff_id,
            str(nonce_digest),
            str(sealed_digest),
            deadline_monotonic,
            deadline_unix_ms,
        )

    def _bound_handoff(
        self, request: Mapping[str, Any], expected: frozenset[str], phase: str
    ) -> _WorkerHandoff:
        values = self._common_handoff_request(request, expected)
        current = self._handoff
        if current is None:
            raise _HandoffRejected("private Worker has no prepared handoff")
        if (
            values[0] != current.handoff_id
            or not hmac.compare_digest(values[1], current.nonce_digest)
            or not hmac.compare_digest(values[2], current.sealed_record_digest)
            or values[3] != current.deadline_monotonic
            or values[4] != current.deadline_unix_ms
        ):
            raise _HandoffRejected("private Worker handoff binding is invalid")
        if current.phase != phase:
            raise _HandoffRejected("private Worker handoff phase is invalid")
        if time.monotonic() >= current.deadline_monotonic or time.time() * 1000 >= current.deadline_unix_ms:
            raise _HandoffAuthorizedFailure("private Worker handoff deadline expired")
        return current

    def _expected_activation(self, handle: PreparedHostHandle) -> dict[str, Any]:
        grant = handle.activation_grant
        if not isinstance(grant, Mapping):
            raise LauncherConfigurationError("activated host has no activation identity")
        return {
            **dict(grant),
            "host": {"pid": handle.host.pid, "birth_id": handle.host_birth_id},
        }

    def _host_control_diagnostic(
        self,
        handle: PreparedHostHandle,
        payload: Mapping[str, Any],
        *,
        stage: str,
        category: str,
        exc: BaseException | None = None,
    ) -> dict[str, Any]:
        operations = frozenset(
            {
                "pause_prepare", "pause_cancel", "rebind_prepare",
                "rebind_commit", "resume_prepare", "resume_commit",
                "handoff_finalize", "handoff_abort", "idle_peek",
            }
        )
        phases = frozenset(
            {
                "owned", "paused", "export_sealed", "adopt_prepared",
                "rebind_committed", "resume_armed", "resumed",
            }
        )
        operation = payload.get("command")
        current = self._handoff
        phase = current.phase if current is not None else "owned"
        diagnostic: dict[str, Any] = {
            "operation": operation if operation in operations else "unknown",
            "handoff_phase": phase if phase in phases else "unknown",
            "stage": stage,
            "exception_category": category,
            "host": {
                "pid": handle.host.pid,
                "birth_id": handle.host_birth_id,
            },
        }
        handoff_id = payload.get("handoff_id")
        if isinstance(handoff_id, str) and 0 < len(handoff_id) <= 256:
            diagnostic["handoff_id"] = handoff_id
        error_number = getattr(exc, "errno", None)
        if (
            not isinstance(error_number, bool)
            and isinstance(error_number, int)
            and -(2**31) <= error_number < 2**31
        ):
            diagnostic["errno"] = error_number
        return diagnostic

    @staticmethod
    def _host_control_exception_category(exc: BaseException) -> str:
        if isinstance(exc, (socket.timeout, TimeoutError)):
            return "timeout"
        if isinstance(exc, BrokenPipeError):
            return "broken_pipe"
        if isinstance(exc, ConnectionResetError):
            return "connection_reset"
        if isinstance(exc, ConnectionError):
            return "connection_error"
        if isinstance(exc, LauncherConfigurationError):
            detail = str(exc)
            if detail == "local Worker control channel closed":
                return "eof"
            if detail in {
                "local Worker control frame is too large",
                "local Worker control channel carried multiple frames",
                "local Worker control frame is malformed",
                "local Worker control frame must be an object",
                "local Worker control frame is not canonical",
                "private control value is not canonical JSON",
            }:
                return "framing"
            return "configuration"
        if isinstance(exc, EOFError):
            return "eof"
        return "os_error"

    def _host_rpc(
        self,
        handle: PreparedHostHandle,
        payload: Mapping[str, Any],
        *,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        self._assert_live(handle)
        timeout: float | None = None
        if deadline_monotonic is not None:
            timeout = deadline_monotonic - time.monotonic()
            if timeout <= 0:
                raise _HandoffAuthorizedFailure("private Worker handoff deadline expired")
        handle.host_control.settimeout(timeout)
        try:
            try:
                _send_private_frame(handle.host_control, payload)
            except (socket.timeout, TimeoutError) as exc:
                raise _HandoffAuthorizedFailure(
                    "GenericPackHost handoff acknowledgement timed out",
                    host_control_diagnostic=self._host_control_diagnostic(
                        handle,
                        payload,
                        stage="send",
                        category=self._host_control_exception_category(exc),
                        exc=exc,
                    ),
                ) from exc
            except (
                BrokenPipeError, ConnectionError, OSError,
                LauncherConfigurationError,
            ) as exc:
                raise _HandoffAuthorizedFailure(
                    "GenericPackHost control channel failed",
                    host_control_diagnostic=self._host_control_diagnostic(
                        handle,
                        payload,
                        stage="send",
                        category=self._host_control_exception_category(exc),
                        exc=exc,
                    ),
                ) from exc
            try:
                response = _receive_private_frame(handle.host_control)
            except (socket.timeout, TimeoutError) as exc:
                raise _HandoffAuthorizedFailure(
                    "GenericPackHost handoff acknowledgement timed out",
                    host_control_diagnostic=self._host_control_diagnostic(
                        handle,
                        payload,
                        stage="receive",
                        category=self._host_control_exception_category(exc),
                        exc=exc,
                    ),
                ) from exc
            except (
                BrokenPipeError, ConnectionError, OSError,
                LauncherConfigurationError,
            ) as exc:
                raise _HandoffAuthorizedFailure(
                    "GenericPackHost control channel failed",
                    host_control_diagnostic=self._host_control_diagnostic(
                        handle,
                        payload,
                        stage="receive",
                        category=self._host_control_exception_category(exc),
                        exc=exc,
                    ),
                ) from exc
        finally:
            try:
                handle.host_control.settimeout(None)
            except OSError:
                pass
        return response

    def _verify_host_ack(
        self,
        handle: PreparedHostHandle,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        *,
        statuses: frozenset[str],
        snapshot: bool = False,
        registration: bool = False,
    ) -> dict[str, Any]:
        fields = {
            "version", "command", "handoff_id", "nonce_digest", "status",
            "host", "phase", "ack_sha256",
        }
        active_refusal = (
            request.get("command") == "pause_prepare"
            and response.get("status") == "active_work"
        )
        if snapshot or active_refusal:
            fields.update(("activation", "registered_state"))
        if active_refusal:
            fields.add("inflight_claim_iterations")
        if request.get("command") in {"rebind_prepare", "rebind_commit"}:
            fields.add("credential_generation")
        if registration:
            fields.add("registration")
        if not isinstance(response, Mapping) or set(response) != fields:
            raise _HandoffAuthorizedFailure("GenericPackHost acknowledgement has an invalid shape")
        host = response.get("host")
        host_matches = (
            isinstance(host, Mapping)
            and set(host) == {"pid", "birth_id"}
            and type(host.get("pid")) is int
            and host.get("pid") == handle.host.pid
            and _process_birth_identities_match(
                handle.host_birth_id, host.get("birth_id")
            )
        )
        if (
            response.get("version") != HOST_CONTROL_VERSION
            or response.get("command") != f"{request['command']}_ack"
            or response.get("handoff_id") != request.get("handoff_id")
            or response.get("nonce_digest") != request.get("nonce_digest")
            or response.get("status") not in statuses
            or not host_matches
        ):
            raise _HandoffAuthorizedFailure("GenericPackHost acknowledgement binding is invalid")
        expected_phase = {
            ("pause_prepare", "paused"): "PAUSED",
            ("pause_prepare", "active_work"): "ACTIVE",
            ("pause_cancel", "pause_cancelled"): "ACTIVE",
            ("rebind_prepare", "rebind_prepared"): "REBIND_PREPARED",
            ("rebind_commit", "rebind_committed"): "REBIND_COMMITTED",
            ("resume_prepare", "resume_prepared"): "RESUME_PREPARED",
            ("resume_commit", "resumed"): "RESUMED",
            ("handoff_finalize", "adopted"): "ADOPTED",
            ("handoff_abort", "aborted"): "ABORTED",
        }.get((str(request.get("command")), str(response.get("status"))))
        if response.get("phase") != expected_phase:
            raise _HandoffAuthorizedFailure("GenericPackHost acknowledgement phase is invalid")
        claimed = response.get("ack_sha256")
        without_digest = {name: value for name, value in response.items() if name != "ack_sha256"}
        if not isinstance(claimed, str) or not hmac.compare_digest(claimed, _sha256_json(without_digest)):
            raise _HandoffAuthorizedFailure("GenericPackHost acknowledgement digest is invalid")
        if snapshot or active_refusal:
            if response.get("activation") != self._expected_activation(handle):
                raise _HandoffAuthorizedFailure("GenericPackHost activation identity changed")
            _registered_state(response.get("registered_state"))
        if active_refusal:
            inflight = response.get("inflight_claim_iterations")
            if isinstance(inflight, bool) or not isinstance(inflight, int) or inflight < 1:
                raise _HandoffAuthorizedFailure(
                    "GenericPackHost active-work count is invalid"
                )
        if request.get("command") in {"rebind_prepare", "rebind_commit"}:
            if response.get("credential_generation") != request.get("credential_generation", self._handoff.credential_generation if self._handoff else None):
                raise _HandoffAuthorizedFailure(
                    "GenericPackHost credential generation changed"
                )
        if registration:
            value = _strict_object(
                response.get("registration"),
                frozenset(
                    {
                        "runtime_registration",
                        "withdrawn_capabilities",
                    }
                ),
                "GenericPackHost registration acknowledgement",
            )
            runtime_registration = _strict_object(
                value["runtime_registration"],
                frozenset({"canonical_bytes", "sha256"}),
                "GenericPackHost Runtime registration receipt",
            )
            canonical_bytes = runtime_registration["canonical_bytes"]
            if (
                isinstance(canonical_bytes, bool)
                or not isinstance(canonical_bytes, int)
                or canonical_bytes < 1
                or canonical_bytes > (1 << 30)
                or not _is_sha256(runtime_registration["sha256"])
            ):
                raise _HandoffAuthorizedFailure(
                    "GenericPackHost Runtime registration receipt is invalid"
                )
            withdrawn = value["withdrawn_capabilities"]
            if (
                not isinstance(withdrawn, list)
                or any(not isinstance(item, str) or not item for item in withdrawn)
                or withdrawn != sorted(withdrawn)
            ):
                raise _HandoffAuthorizedFailure(
                    "GenericPackHost withdrawn capability acknowledgement is invalid"
                )
        return dict(response)

    def _forward_host(
        self,
        handle: PreparedHostHandle,
        payload: Mapping[str, Any],
        *,
        statuses: frozenset[str],
        deadline_monotonic: float | None = None,
        snapshot: bool = False,
        registration: bool = False,
    ) -> dict[str, Any]:
        response = self._host_rpc(
            handle, payload, deadline_monotonic=deadline_monotonic
        )
        try:
            return self._verify_host_ack(
                handle,
                payload,
                response,
                statuses=statuses,
                snapshot=snapshot,
                registration=registration,
            )
        except _HandoffAuthorizedFailure as exc:
            if exc.host_control_diagnostic is None:
                exc.host_control_diagnostic = self._host_control_diagnostic(
                    handle,
                    payload,
                    stage="validation",
                    category="ack_validation",
                )
            raise

    def _validate_handoff_export(
        self,
        handle: PreparedHostHandle,
        current: _WorkerHandoff,
        value: object,
    ) -> dict[str, Any]:
        export = _strict_object(
            value,
            frozenset(
                {"receipt", "identity", "credential_generation", "registered_state"}
            ),
            "handoff export",
        )
        if _contains_raw_nonce(export):
            raise _HandoffRejected("private Worker handoff export contains a raw nonce")
        receipt = export["receipt"]
        identity = export["identity"]
        if not isinstance(receipt, Mapping) or not isinstance(identity, Mapping):
            raise _HandoffRejected("private Worker handoff export graph is invalid")
        identity_value = dict(identity)
        expected_receipt = {
            "version": RECEIPT_VERSION,
            **identity_value,
            "executor_incarnation": (handle.activation_grant or {}).get(
                "executor_incarnation"
            ),
        }
        if dict(receipt) != expected_receipt:
            raise _HandoffRejected("private Worker handoff receipt is invalid")
        if (
            identity_value.get("evidence_digest") != current.receipt_evidence_digest
            or export["credential_generation"] != current.credential_generation
            or _registered_state(export["registered_state"]) != current.registered_state
        ):
            raise _HandoffRejected("private Worker handoff export evidence is invalid")
        engine = handle.engine_report
        expected_processes = {
            "worker": {"pid": os.getpid(), "birth_id": self._birth(os.getpid())},
            "host": {"pid": handle.host.pid, "birth_id": handle.host_birth_id},
            "engine": {
                "pid": int(engine.get("pid", 0)),
                "birth_id": str(engine.get("process_birth_id", "")),
            },
            "engine_listener": {
                "pid": int(engine.get("comfy_pid", 0)),
                "birth_id": str(engine.get("comfy_process_birth_id", "")),
            },
        }
        for name, expected in expected_processes.items():
            process = identity_value.get(name)
            if (
                not isinstance(process, Mapping)
                or process.get("pid") != expected["pid"]
                or process.get("birth_id") != expected["birth_id"]
            ):
                raise _HandoffRejected("private Worker handoff export graph is invalid")
        return export

    def _cached_handoff_ack(
        self, request: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bool] | None:
        current = self._handoff
        command = request.get("command")
        if current is None or not isinstance(command, str):
            return None
        same_identity = (
            request.get("handoff_id") == current.handoff_id
            and request.get("nonce_digest") == current.nonce_digest
            and request.get("sealed_record_digest") == current.sealed_record_digest
            and (
                command == "handoff_seal"
                or (
                    request.get("deadline_monotonic")
                    == current.deadline_monotonic
                    and request.get("deadline_unix_ms") == current.deadline_unix_ms
                )
            )
        )
        if not same_identity:
            return None
        acknowledgements = current.acknowledgements or {}
        cached = acknowledgements.get(command)
        if cached is None:
            return None
        if command == "handoff_seal":
            raise _HandoffRejected(
                "private Worker handoff seal authority is already consumed"
            )
        replay_phases = {
            "handoff_prepare": "paused",
            "handoff_seal": "export_sealed",
            "handoff_adopt": "adopt_prepared",
            "handoff_commit": "rebind_committed",
            "resume_prepare": "resume_armed",
            "resume_commit": "resumed",
        }
        if replay_phases.get(command) != current.phase:
            raise _HandoffRejected("private Worker handoff replay phase is invalid")
        if (
            time.monotonic() >= current.deadline_monotonic
            or time.time() * 1000 >= current.deadline_unix_ms
        ):
            raise _HandoffAuthorizedFailure("private Worker handoff deadline expired")
        digest = _sha256_json(dict(request))
        if cached[0] != digest:
            raise _HandoffRejected("private Worker handoff replay changed")
        return dict(cached[1]), False

    def _remember_handoff_ack(
        self, request: Mapping[str, Any], response: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        current = self._handoff
        if current is not None:
            if current.acknowledgements is None:
                current.acknowledgements = {}
            current.acknowledgements[str(request.get("command"))] = (
                _sha256_json(dict(request)),
                dict(response),
            )
        return response, False

    def _finalized_handoff_ack(
        self, request: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bool] | None:
        handoff_id = request.get("handoff_id")
        tombstone = next(
            (
                item
                for item in reversed(self._finalized_handoff_acks)
                if item.handoff_id == handoff_id
            ),
            None,
        )
        if tombstone is None:
            return None
        if (
            self._handoff is None
            and request.get("command") == "handoff_finalize"
            and hmac.compare_digest(
                _sha256_json(dict(request)), tombstone.request_digest
            )
        ):
            return dict(tombstone.response), False
        raise _HandoffRejected("private Worker handoff is finalized")

    def _remember_finalized_handoff_ack(
        self, request: Mapping[str, Any], response: Mapping[str, Any]
    ) -> None:
        handoff_id = str(request["handoff_id"])
        self._finalized_handoff_acks = [
            item
            for item in self._finalized_handoff_acks
            if item.handoff_id != handoff_id
        ]
        self._finalized_handoff_acks.append(
            _FinalizedHandoffAck(
                handoff_id=handoff_id,
                request_digest=_sha256_json(dict(request)),
                response=dict(response),
            )
        )
        del self._finalized_handoff_acks[:-8]

    def handoff_command(
        self, handle: object, request: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        if not isinstance(handle, PreparedHostHandle) or handle is not self._active:
            raise _HandoffRejected("prepared host handle is not owned by this Worker")
        if not handle.activated:
            raise _HandoffRejected("GenericPackHost is not activated")
        command = request.get("command")
        finalized = self._finalized_handoff_ack(request)
        if finalized is not None:
            return finalized
        cached = self._cached_handoff_ack(request)
        if cached is not None:
            return cached
        if command == "handoff_prepare":
            values = self._common_handoff_request(
                request,
                frozenset(
                    {
                        "old_owner", "old_runtime",
                        "receipt_evidence_digest", "credential_generation",
                        "sealed_record",
                    }
                ),
            )
            if self._handoff is not None:
                raise _HandoffRejected("private Worker handoff is already prepared")
            if time.monotonic() >= values[3] or time.time() * 1000 >= values[4]:
                raise _HandoffRejected("private Worker handoff deadline expired")
            old_runtime = _runtime_identity(request.get("old_runtime"), "old Runtime")
            old_owner = _runtime_owner(request.get("old_owner"), "old owner")
            if (
                self._runtime_owner_identity is not None
                and old_owner != self._runtime_owner_identity
            ):
                raise _HandoffRejected("private Worker owner A is stale")
            sealed_record = _sealed_handoff_record(
                request.get("sealed_record"),
                handoff_id=values[0],
                nonce_digest=values[1],
                sealed_record_digest=values[2],
                old_owner=old_owner,
                old_runtime=old_runtime,
                deadline_monotonic=values[3],
                deadline_unix_ms=values[4],
            )
            credential = _credential_generation(request.get("credential_generation"))
            receipt_digest = request.get("receipt_evidence_digest")
            if not _is_sha256(receipt_digest):
                raise _HandoffRejected("handoff receipt evidence digest is invalid")
            if receipt_digest != (handle.activation_grant or {}).get("evidence_digest"):
                raise _HandoffRejected("handoff receipt evidence does not match activation")
            host_request = {
                "version": HOST_CONTROL_VERSION,
                "command": "pause_prepare",
                "handoff_id": values[0],
                "nonce_digest": values[1],
                "deadline_monotonic": values[3],
                "deadline_unix_ms": values[4],
                "old_runtime": old_runtime,
                "old_owner": old_owner,
            }
            host_ack = self._forward_host(
                handle,
                host_request,
                statuses=frozenset({"paused", "active_work"}),
                deadline_monotonic=values[3],
                snapshot=True,
            )
            state = _registered_state(host_ack["registered_state"])
            if state["runtime"] != old_runtime:
                raise _HandoffAuthorizedFailure("GenericPackHost old Runtime state changed")
            if host_ack["status"] == "active_work":
                return self._ack_payload(
                    {
                        "version": CONTROL_VERSION,
                        "command": "handoff_prepare_ack",
                        "handoff_id": values[0],
                        "status": "active_work",
                        "nonce_digest": values[1],
                        "sealed_record_digest": values[2],
                        "host_ack": host_ack,
                        "worker_phase": "owned",
                    }
                ), False
            self._handoff = _WorkerHandoff(
                handoff_id=values[0],
                nonce_digest=values[1],
                sealed_record_digest=values[2],
                deadline_monotonic=values[3],
                deadline_unix_ms=values[4],
                old_runtime=old_runtime,
                receipt_evidence_digest=str(receipt_digest),
                credential_generation=credential,
                registered_state=state,
                old_owner=old_owner,
                sealed_record=sealed_record,
                acknowledgements={},
            )
            self._runtime_owner_identity = old_owner
            response = self._ack_payload(
                {
                    "version": CONTROL_VERSION,
                    "command": "handoff_prepare_ack",
                    "handoff_id": values[0],
                    "status": "prepared",
                    "nonce_digest": values[1],
                    "sealed_record_digest": values[2],
                    "host_ack": host_ack,
                    "worker_phase": "paused",
                }
            )
            return self._remember_handoff_ack(request, response)

        if command == "handoff_seal":
            expected = {
                "version", "command", "handoff_id", "nonce", "nonce_digest",
                "sealed_record_digest", "export_sealed_digest",
                "export_record_digest", "export", "old_owner",
            }
            if set(request) != expected or request.get("version") != CONTROL_VERSION:
                raise _HandoffRejected(
                    "private Worker handoff seal request has an invalid shape"
                )
            current = self._handoff
            if current is None:
                raise _HandoffRejected("private Worker has no prepared handoff")
            if (
                current.phase != "paused"
                or request.get("handoff_id") != current.handoff_id
                or request.get("nonce_digest") != current.nonce_digest
                or request.get("sealed_record_digest")
                != current.sealed_record_digest
            ):
                raise _HandoffRejected("private Worker handoff seal binding is invalid")
            if (
                time.monotonic() >= current.deadline_monotonic
                or time.time() * 1000 >= current.deadline_unix_ms
            ):
                raise _HandoffAuthorizedFailure("private Worker handoff deadline expired")
            nonce = request.get("nonce")
            if (
                not isinstance(nonce, str)
                or len(nonce) < 32
                or not hmac.compare_digest(_nonce_digest(nonce), current.nonce_digest)
                or _runtime_owner(request.get("old_owner"), "old owner")
                != current.old_owner
            ):
                raise _HandoffRejected("private Worker handoff seal authority is invalid")
            export_sealed_digest = request.get("export_sealed_digest")
            export_record_digest = request.get("export_record_digest")
            if not _is_sha256(export_sealed_digest) or not _is_sha256(
                export_record_digest
            ):
                raise _HandoffRejected("private Worker handoff export digest is invalid")
            export = self._validate_handoff_export(
                handle, current, request.get("export")
            )
            expected_export_seal = _sha256_json(
                {
                    "version": HANDOFF_EXPORT_SEAL_VERSION,
                    "sealed_record_digest": current.sealed_record_digest,
                    "nonce_digest": current.nonce_digest,
                    "export": export,
                }
            )
            if not hmac.compare_digest(
                str(export_sealed_digest), expected_export_seal
            ):
                raise _HandoffRejected("private Worker handoff export seal is invalid")
            bound_record = {
                key: item
                for key, item in current.sealed_record.items()
                if key != "record_digest"
            }
            bound_record["export"] = export
            bound_record["export_sealed_digest"] = str(export_sealed_digest)
            if not hmac.compare_digest(
                str(export_record_digest), _sha256_json(bound_record)
            ):
                raise _HandoffRejected(
                    "private Worker handoff export record digest is invalid"
                )
            current.export_sealed_digest = str(export_sealed_digest)
            current.export_record_digest = str(export_record_digest)
            current.export_digest = _sha256_json(export)
            current.phase = "export_sealed"
            response = self._ack_payload(
                {
                    "version": CONTROL_VERSION,
                    "command": "handoff_seal_ack",
                    "handoff_id": current.handoff_id,
                    "status": "sealed",
                    "nonce_digest": current.nonce_digest,
                    "sealed_record_digest": current.sealed_record_digest,
                    "host_ack": {
                        "status": "export_sealed",
                        "host": {
                            "pid": handle.host.pid,
                            "birth_id": handle.host_birth_id,
                        },
                    },
                    "worker_phase": "export_sealed",
                }
            )
            return self._remember_handoff_ack(request, response)

        if command == "handoff_adopt":
            current = self._bound_handoff(
                request,
                frozenset(
                    {
                        "nonce", "request_id", "old_runtime", "new_runtime", "endpoint",
                        "credential_file", "credential_generation", "receipt_evidence_digest",
                        "executor_incarnation", "registered_state", "old_owner", "new_owner",
                        "export_sealed_digest", "export_record_digest",
                        "adopter_record_digest",
                    }
                ),
                "export_sealed",
            )
            nonce = request.get("nonce")
            if (
                not isinstance(nonce, str)
                or len(nonce) < 32
                or not hmac.compare_digest(_nonce_digest(nonce), current.nonce_digest)
            ):
                raise _HandoffRejected("private Worker handoff nonce is invalid")
            request_id = request.get("request_id")
            if (
                not isinstance(request_id, str)
                or request_id != current.handoff_id
                or len(request_id) > 256
            ):
                raise _HandoffRejected("private Worker adopter request is invalid")
            old_runtime = _runtime_identity(request.get("old_runtime"), "old Runtime")
            new_runtime = _runtime_identity(request.get("new_runtime"), "new Runtime")
            old_owner = _runtime_owner(request.get("old_owner"), "old owner")
            new_owner = _runtime_owner(request.get("new_owner"), "new owner")
            generation = _credential_generation(request.get("credential_generation"))
            registered = _registered_state(request.get("registered_state"))
            credential_file = request.get("credential_file")
            incarnation = request.get("executor_incarnation")
            adopter_record_digest = request.get("adopter_record_digest")
            runtime_transition_valid = (
                new_runtime["endpoint"] == old_runtime["endpoint"]
                and new_runtime["protocol"] == old_runtime["protocol"]
                and new_runtime["schema_digest"] == old_runtime["schema_digest"]
                and new_runtime["runtime_epoch"] > old_runtime["runtime_epoch"]
                and new_runtime["runtime_instance_id"] != old_runtime["runtime_instance_id"]
                and new_runtime["runtime_session_id"] != old_runtime["runtime_session_id"]
            )
            if (
                old_runtime != current.old_runtime
                or old_owner != current.old_owner
                or new_owner == current.old_owner
                or request.get("export_sealed_digest")
                != current.export_sealed_digest
                or request.get("export_record_digest") != current.export_record_digest
                or not _is_sha256(adopter_record_digest)
                or adopter_record_digest == current.export_record_digest
                or not runtime_transition_valid
                or generation != current.credential_generation
                or registered != current.registered_state
                or request.get("receipt_evidence_digest") != current.receipt_evidence_digest
                or request.get("endpoint") != new_runtime["endpoint"]
                or credential_file != str(self.config.credential_file)
                or not isinstance(incarnation, str)
                or not incarnation
                or incarnation != (handle.activation_grant or {}).get("executor_incarnation")
            ):
                raise _HandoffRejected("private Worker adopter evidence is invalid")
            host_request = {
                "version": HOST_CONTROL_VERSION,
                "command": "rebind_prepare",
                "handoff_id": current.handoff_id,
                "nonce_digest": current.nonce_digest,
                "deadline_monotonic": current.deadline_monotonic,
                "deadline_unix_ms": current.deadline_unix_ms,
                "new_runtime": new_runtime,
                "credential_file": str(self.config.credential_file),
                "credential_generation": generation,
                "registered_state": registered,
                "old_owner": current.old_owner,
                "new_owner": new_owner,
            }
            host_ack = self._forward_host(
                handle,
                host_request,
                statuses=frozenset({"rebind_prepared"}),
                deadline_monotonic=current.deadline_monotonic,
                snapshot=True,
            )
            prospective = _registered_state(host_ack["registered_state"])
            if prospective["runtime"] != new_runtime:
                raise _HandoffAuthorizedFailure("GenericPackHost did not preview the new Runtime")
            current.phase = "adopt_prepared"
            current.adopter_request_id = request_id
            current.new_runtime = new_runtime
            current.new_owner = new_owner
            current.adopter_record_digest = str(adopter_record_digest)
            current.registered_state = prospective
            current.nonce_consumed = True
            response = self._ack_payload(
                {
                    "version": CONTROL_VERSION,
                    "command": "handoff_adopt_ack",
                    "handoff_id": current.handoff_id,
                    "status": "prepared",
                    "nonce_digest": current.nonce_digest,
                    "sealed_record_digest": current.sealed_record_digest,
                    "host_ack": host_ack,
                    "worker_phase": "adopt_prepared",
                }
            )
            return self._remember_handoff_ack(request, response)

        if command == "handoff_commit":
            current = self._bound_handoff(
                request,
                frozenset(
                    {"new_owner", "new_runtime", "credential_generation", "registered_state"}
                ),
                "adopt_prepared",
            )
            if (
                _runtime_owner(request.get("new_owner"), "new owner") != current.new_owner
                or _runtime_identity(request.get("new_runtime"), "new Runtime")
                != current.new_runtime
                or _credential_generation(request.get("credential_generation"))
                != current.credential_generation
                or _registered_state(request.get("registered_state"))
                != current.registered_state
            ):
                raise _HandoffRejected("private Worker commit evidence is invalid")
            host_request = {
                "version": HOST_CONTROL_VERSION,
                "command": "rebind_commit",
                "handoff_id": current.handoff_id,
                "nonce_digest": current.nonce_digest,
                "new_owner": current.new_owner,
            }
            host_ack = self._forward_host(
                handle,
                host_request,
                statuses=frozenset({"rebind_committed"}),
                deadline_monotonic=current.deadline_monotonic,
                snapshot=True,
                registration=True,
            )
            rebound = _registered_state(host_ack["registered_state"])
            if (
                rebound != current.registered_state
                or rebound["runtime"] != current.new_runtime
            ):
                raise _HandoffAuthorizedFailure("GenericPackHost did not bind the new Runtime")
            current.registered_state = rebound
            current.phase = "rebind_committed"
            response_status = "committed"
            response_phase = "rebind_committed"
        elif command == "resume_prepare":
            current = self._bound_handoff(
                request, frozenset({"new_owner", "new_runtime"}), "rebind_committed"
            )
            if (
                _runtime_owner(request.get("new_owner"), "new owner") != current.new_owner
                or _runtime_identity(request.get("new_runtime"), "new Runtime")
                != current.new_runtime
            ):
                raise _HandoffRejected("private Worker resume Runtime is invalid")
            host_request = {
                "version": HOST_CONTROL_VERSION,
                "command": "resume_prepare",
                "handoff_id": current.handoff_id,
                "nonce_digest": current.nonce_digest,
                "new_owner": current.new_owner,
            }
            host_ack = self._forward_host(
                handle,
                host_request,
                statuses=frozenset({"resume_prepared"}),
                deadline_monotonic=current.deadline_monotonic,
            )
            current.phase = "resume_armed"
            response_status = "prepared"
            response_phase = "resume_armed"
        elif command == "resume_commit":
            current = self._bound_handoff(
                request, frozenset({"new_owner", "new_runtime"}), "resume_armed"
            )
            if (
                _runtime_owner(request.get("new_owner"), "new owner") != current.new_owner
                or _runtime_identity(request.get("new_runtime"), "new Runtime")
                != current.new_runtime
            ):
                raise _HandoffRejected("private Worker resume Runtime is invalid")
            host_request = {
                "version": HOST_CONTROL_VERSION,
                "command": "resume_commit",
                "handoff_id": current.handoff_id,
                "nonce_digest": current.nonce_digest,
                "new_owner": current.new_owner,
            }
            host_ack = self._forward_host(
                handle,
                host_request,
                statuses=frozenset({"resumed"}),
                deadline_monotonic=current.deadline_monotonic,
            )
            current.phase = "resumed"
            response_status = "committed"
            response_phase = "resumed"
        elif command == "handoff_finalize":
            current = self._bound_handoff(
                request, frozenset({"new_owner"}), "resumed"
            )
            new_owner = _runtime_owner(request.get("new_owner"), "new owner")
            if new_owner != current.new_owner:
                raise _HandoffRejected("private Worker finalizer identity is invalid")
            host_request = {
                "version": HOST_CONTROL_VERSION,
                "command": "handoff_finalize",
                "handoff_id": current.handoff_id,
                "nonce_digest": current.nonce_digest,
                "new_owner": current.new_owner,
            }
            host_ack = self._forward_host(
                handle,
                host_request,
                statuses=frozenset({"adopted"}),
                deadline_monotonic=current.deadline_monotonic,
            )
            response = self._ack_payload(
                {
                    "version": CONTROL_VERSION,
                    "command": "handoff_finalize_ack",
                    "handoff_id": current.handoff_id,
                    "status": "finalized",
                    "nonce_digest": current.nonce_digest,
                    "sealed_record_digest": current.sealed_record_digest,
                    "host_ack": host_ack,
                    "worker_phase": "finalized",
                }
            )
            self._runtime_owner_identity = new_owner
            self._remember_finalized_handoff_ack(request, response)
            self._handoff = None
            return response, False
        elif command == "handoff_abort":
            phase = str(self._handoff.phase if self._handoff else "")
            owner_field = (
                "old_owner"
                if phase in {"paused", "export_sealed"}
                else "new_owner"
            )
            current = self._bound_handoff(
                request, frozenset({"reason_code", owner_field}), phase
            )
            expected_owner = current.old_owner if owner_field == "old_owner" else current.new_owner
            if _runtime_owner(request.get(owner_field), owner_field) != expected_owner:
                raise _HandoffRejected("private Worker abort owner is invalid")
            reason = request.get("reason_code")
            if not isinstance(reason, str) or not reason or len(reason) > 256:
                raise _HandoffRejected("private Worker abort reason is invalid")
            host_command = (
                "pause_cancel"
                if current.phase in {"paused", "export_sealed"}
                else "handoff_abort"
            )
            host_request = {
                "version": HOST_CONTROL_VERSION,
                "command": host_command,
                "handoff_id": current.handoff_id,
                "nonce_digest": current.nonce_digest,
            }
            if host_command == "handoff_abort":
                host_request["reason"] = reason
                statuses = frozenset({"aborted"})
            else:
                host_request["old_owner"] = current.old_owner
                statuses = frozenset({"pause_cancelled"})
            host_ack = self._forward_host(
                handle,
                host_request,
                statuses=statuses,
                deadline_monotonic=current.deadline_monotonic,
            )
            if host_command == "pause_cancel":
                self._handoff = None
                return self._ack_payload(
                    {
                        "version": CONTROL_VERSION,
                        "command": "handoff_abort_ack",
                        "handoff_id": current.handoff_id,
                        "status": "cancelled",
                        "nonce_digest": current.nonce_digest,
                        "sealed_record_digest": current.sealed_record_digest,
                        "host_ack": host_ack,
                        "worker_phase": "owned",
                    }
                ), False
            self.abort(handle)
            return self._ack_payload(
                {
                    "version": CONTROL_VERSION,
                    "command": "handoff_abort_ack",
                    "handoff_id": current.handoff_id,
                    "status": "aborted",
                    "nonce_digest": current.nonce_digest,
                    "sealed_record_digest": current.sealed_record_digest,
                    "host_ack": host_ack,
                    "worker_phase": "aborted",
                }
            ), True
        else:
            raise _HandoffRejected("private Worker handoff command is invalid")

        response = self._ack_payload(
            {
                "version": CONTROL_VERSION,
                "command": f"{command}_ack",
                "handoff_id": current.handoff_id,
                "status": response_status,
                "nonce_digest": current.nonce_digest,
                "sealed_record_digest": current.sealed_record_digest,
                "host_ack": host_ack,
                "worker_phase": response_phase,
            }
        )
        return self._remember_handoff_ack(request, response)

    def abort(self, handle: object) -> None:
        if not isinstance(handle, PreparedHostHandle) or handle is not self._active:
            return
        error: BaseException | None = None
        try:
            if handle.activation.fileno() >= 0:
                handle.activation.close()
            if handle.host_control.fileno() >= 0:
                handle.host_control.close()
            if handle.host.poll() is None:
                identity = handle.host_cleanup_identity
                if not _verify_cleanup_identity(identity):
                    raise LauncherConfigurationError(
                        "prepared host disappeared before safe cleanup"
                    )
                if identity.process_group != identity.pid or identity.session_id != identity.pid:
                    raise LauncherConfigurationError(
                        "prepared host cleanup group is invalid"
                    )
                _signal_owned_group(handle.host, signal.SIGTERM)
                try:
                    handle.host.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if _verify_cleanup_identity(identity):
                        _signal_owned_group(handle.host, signal.SIGKILL)
                    handle.host.wait(timeout=3)
            _stop_owned_vibecomfy_session(handle.engine)
        except BaseException as exc:
            error = exc
        finally:
            self.config.ready_file.unlink(missing_ok=True)
            if handle.readiness_profile is not None:
                handle.readiness_profile.unlink(missing_ok=True)
            handle.closed = True
            self._active = None
            self._handoff = None
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
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
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
    if _canonical_json(value) != encoded:
        raise LauncherConfigurationError("local Worker control frame is not canonical")
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
        "launch_mode": config.launch_mode,
        "source_checkout": str(config.source_checkout) if config.source_checkout is not None else None,
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
        "host_python", "launch_mode", "source_checkout", "pack_root", "runtime_endpoint",
        "credential_file", "support_root", "runtime_instance_id", "ready_file",
        "state_file", "boot_manifest_path", "boot_manifest_hash", "capability_matrix",
        "readiness_profile_path", "readiness_profile_hash",
    }
    if set(value) != expected:
        raise LauncherConfigurationError("private host configuration has an invalid shape")
    launch_mode = value.get("launch_mode")
    if launch_mode not in {"editable", "installed"}:
        raise LauncherConfigurationError("private host launch mode is invalid")
    if launch_mode == "editable" and value.get("source_checkout") is None:
        raise LauncherConfigurationError("editable private host configuration requires source_checkout")
    if launch_mode == "installed" and value.get("source_checkout") is not None:
        raise LauncherConfigurationError("installed private host configuration forbids source_checkout")
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
        custody_owner = _CustodyLaunchOwner("worker")
        try:
            executable = Path(getattr(profile, "worker_executable"))
            worker_root = Path(__file__).resolve().parents[2]
            child_env = dict(self.environ)
            child_env.pop("PYTHONPATH", None)
            worker_cwd = self.config.support_root
            if self.config.launch_mode == "editable":
                child_env["PYTHONPATH"] = str(worker_root)
                worker_cwd = worker_root
            worker = _custodied_popen(
                [
                    str(executable), "-m", "source.runtime.supervisor",
                    "--prepared-control-fd", str(child.fileno()),
                ],
                custody_role="worker",
                custody_owner=custody_owner,
                cwd=str(worker_cwd),
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

    def handoff(self, handle: object, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send one exact v2 handoff command over the retained live descriptor."""
        if not isinstance(handle, PreparedWorkerHandle) or handle is not self._active:
            raise LauncherConfigurationError("prepared Worker handle is not active")
        if request.get("version") != CONTROL_VERSION:
            raise LauncherConfigurationError("prepared Worker handoff version is invalid")
        if handle.closed or handle.worker.poll() is not None:
            raise LauncherConfigurationError("prepared Worker is not alive")
        if self._birth(handle.worker.pid) != handle.worker_birth_id:
            raise LauncherConfigurationError("prepared Worker identity changed")
        _send_private_frame(handle.control, request)
        response = _receive_private_frame(handle.control)
        if response.get("version") != CONTROL_VERSION:
            raise LauncherConfigurationError("prepared Worker control version is invalid")
        if response.get("status") == "error":
            raise LauncherConfigurationError(
                str(response.get("error") or "prepared Worker rejected the handoff")
            )
        claimed = response.get("ack_sha256")
        without_digest = {name: value for name, value in response.items() if name != "ack_sha256"}
        if not isinstance(claimed, str) or not hmac.compare_digest(
            claimed, _sha256_json(without_digest)
        ):
            raise LauncherConfigurationError("prepared Worker handoff acknowledgement is invalid")
        return response


_host_control_diagnostic_for = LocalWorkerPreparerAdapter._host_control_diagnostic
_host_control_exception_category = LocalWorkerPreparerAdapter._host_control_exception_category


def _serve_prepared_worker(descriptor: int) -> int:
    """Run the private Worker side of ``LocalWorkerProcessPreparer``."""

    control = socket.socket(fileno=descriptor)
    control.settimeout(None)
    adapter: LocalWorkerPreparerAdapter | None = None
    handle: PreparedHostHandle | None = None
    try:
        while True:
            try:
                handoff = getattr(adapter, "_handoff", None)
                wait_timeout: float | None = None
                if handoff is not None and handoff.phase != "resumed":
                    remaining = handoff.deadline_monotonic - time.monotonic()
                    if remaining <= 0:
                        raise _HandoffAuthorizedFailure("private Worker handoff deadline expired")
                    wait_timeout = remaining
                watched: list[socket.socket] = [control]
                host_channel = getattr(handle, "host_control", None)
                if isinstance(host_channel, socket.socket) and host_channel.fileno() >= 0:
                    watched.append(host_channel)
                readable, _, _ = select.select(watched, [], [], wait_timeout)
                if not readable:
                    raise _HandoffAuthorizedFailure("private Worker handoff deadline expired")
                if host_channel in readable:
                    try:
                        pending = host_channel.recv(1, socket.MSG_PEEK)
                    except OSError as exc:
                        raise _HandoffAuthorizedFailure(
                            "GenericPackHost control channel failed",
                            host_control_diagnostic=_host_control_diagnostic_for(
                                adapter,
                                handle,
                                {"command": "idle_peek"},
                                stage="peek",
                                category=_host_control_exception_category(exc),
                                exc=exc,
                            ),
                        ) from exc
                    if not pending:
                        exc = EOFError("host-control EOF")
                        raise _HandoffAuthorizedFailure(
                            "GenericPackHost control channel closed",
                            host_control_diagnostic=_host_control_diagnostic_for(
                                adapter,
                                handle,
                                {"command": "idle_peek"},
                                stage="peek",
                                category=_host_control_exception_category(exc),
                                exc=exc,
                            ),
                        ) from exc
                    raise _HandoffAuthorizedFailure(
                        "GenericPackHost control channel sent an unsolicited frame",
                        host_control_diagnostic=adapter._host_control_diagnostic(
                            handle,
                            {"command": "idle_peek"},
                            stage="peek",
                            category="unsolicited_frame",
                        ),
                    )
                control.settimeout(None)
                request = _receive_private_frame(control)
                if request.get("version") != CONTROL_VERSION:
                    raise LauncherConfigurationError("private Worker control version is invalid")
                command = request.get("command")
                if command == "prepare":
                    if set(request) != {
                        "version", "command", "operation_id", "channel_id", "profile", "config"
                    }:
                        raise LauncherConfigurationError("private Worker prepare request has an invalid shape")
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
                    if set(request) != {"version", "command"}:
                        raise LauncherConfigurationError("private Worker report request has an invalid shape")
                    response = {
                        "version": CONTROL_VERSION,
                        "status": "ok",
                        "report": dict(adapter.report(handle)),
                    }
                elif command == "activate" and adapter is not None and handle is not None:
                    if set(request) != {"version", "command", "grant"}:
                        raise LauncherConfigurationError("private Worker activation request has an invalid shape")
                    grant = request.get("grant")
                    if not isinstance(grant, Mapping):
                        raise LauncherConfigurationError("private Worker activation grant is invalid")
                    adapter.activate(handle, grant)
                    response = {"version": CONTROL_VERSION, "status": "ok"}
                elif command == "reconnect" and adapter is not None and handle is not None:
                    if set(request) != {"version", "command", "receipt"}:
                        raise LauncherConfigurationError("private Worker reconnect request has an invalid shape")
                    receipt = request.get("receipt")
                    if not isinstance(receipt, Mapping):
                        raise LauncherConfigurationError("private Worker reconnect receipt is invalid")
                    response = {
                        "version": CONTROL_VERSION,
                        "status": "ok",
                        "reconnected": adapter.reconnect(receipt) is handle,
                    }
                elif command == "abort" and adapter is not None and handle is not None:
                    if set(request) != {"version", "command"}:
                        raise LauncherConfigurationError("private Worker abort request has an invalid shape")
                    adapter.abort(handle)
                    _send_private_frame(
                        control, {"version": CONTROL_VERSION, "status": "ok"}
                    )
                    return 0
                elif (
                    command in {
                        "handoff_prepare", "handoff_seal", "handoff_adopt",
                        "handoff_commit", "resume_prepare", "resume_commit",
                        "handoff_finalize", "handoff_abort",
                    }
                    and adapter is not None
                    and handle is not None
                ):
                    response, should_exit = adapter.handoff_command(handle, request)
                    _send_private_frame(control, response)
                    if should_exit:
                        return 0
                    continue
                else:
                    raise LauncherConfigurationError("private Worker command is invalid")
                _send_private_frame(control, response)
            except _HandoffAuthorizedFailure as exc:
                # Publish the bounded refusal before cleanup.  Abort may take
                # longer than the Runtime control timeout; delaying this frame
                # until after abort makes the authenticated peer observe only
                # EOF and loses the credential-safe failure classification.
                try:
                    _send_private_frame(
                        control,
                        _authorized_handoff_error_response(exc, command),
                    )
                except BaseException:
                    pass
                if adapter is not None and handle is not None:
                    try:
                        adapter.abort(handle)
                    except BaseException:
                        pass
                return 78
            except _HandoffRejected as exc:
                _send_private_frame(
                    control,
                    {
                        "version": CONTROL_VERSION,
                        "status": "error",
                        "error": "prepared Worker rejected the handoff",
                        "error_code": _bounded_handoff_error_code(exc),
                        "error_stage": str(command or "control"),
                    },
                )
            except LauncherConfigurationError as exc:
                _send_private_frame(
                    control,
                    {
                        "version": CONTROL_VERSION,
                        "status": "error",
                        "error": "prepared Worker rejected the handoff",
                        "error_code": _bounded_handoff_error_code(exc),
                        "error_stage": str(command or "control"),
                    },
                )
    except (BrokenPipeError, ConnectionError, OSError, socket.timeout):
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
    custody_owner = _CustodyLaunchOwner("generic_pack_host")

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
            child = _custodied_popen(
                argv,
                custody_role="generic_pack_host",
                custody_owner=custody_owner,
                cwd=str(_host_working_directory(config)),
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
                    _signal_owned_group(child, signal.SIGTERM)
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


if __name__ == "__main__":
    raise SystemExit(main())
