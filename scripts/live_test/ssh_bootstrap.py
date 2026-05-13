"""SSH-side worker bootstrap helpers shared by the live-test variants."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
import time
from typing import Literal

import scripts.live_test as live_test_pkg


APT_INSTALL_PACKAGES = (
    "python3.10-venv",
    "python3.10-dev",
    "build-essential",
    "ffmpeg",
    "git",
    "curl",
    "wget",
)
PROCESS_SCAN_COMMAND = (
    r"ps -eo pid=,args= | awk '/run_worker[.]py|python[^ ]* .*worker[.]py|source[.]runtime[.]worker/ {print}'"
)
KILL_COMMAND = (
    "pkill -f 'run_worker[.]py'; "
    "pkill -f 'python worker[.]py'; "
    "pkill -f 'python[^ ]* .*worker[.]py'; "
    "pkill -f 'source[.]runtime[.]worker'; "
    "sleep 5; "
    "pkill -9 -f 'run_worker[.]py' || true; "
    "pkill -9 -f 'python worker[.]py' || true; "
    "pkill -9 -f 'python[^ ]* .*worker[.]py' || true; "
    "pkill -9 -f 'source[.]runtime[.]worker' || true; "
    "sleep 2"
)


@dataclass(frozen=True)
class WorkerProcessInfo:
    family: Literal["supervisor", "direct"]
    cmdline: list[str]
    pid: int


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def _resolve_attention_profile(raw: str | None = None) -> str:
    value = (
        raw
        or os.environ.get("REIGH_VIBECOMFY_ATTENTION_PROFILE")
        or os.environ.get("VIBECOMFY_ATTENTION_PROFILE")
        or ""
    ).strip().lower()
    if value in {"", "default", "portable", "sdpa"}:
        return "portable"
    if value in {"optimized", "sage", "sageattn", "sageattention"}:
        return "sage"
    raise ValueError("VibeComfy attention profile must be 'portable' or 'sage'")


def _sageattention_install_block(python_path: str, *, uv_path: str | None = None) -> str:
    py = _quote(python_path)
    uv = _quote(uv_path or _uv_for_python(python_path))
    return (
        "rm -rf /tmp/sageattention\n"
        "git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/sageattention\n"
        f"{uv} pip install --python {py} --no-build-isolation /tmp/sageattention\n"
        f"{py} - <<'PY'\n"
        "import sageattention\n"
        "if not callable(getattr(sageattention, 'sageattn', None)):\n"
        "    raise RuntimeError('sageattention import succeeded but sageattn is missing')\n"
        "print('sageattention verified')\n"
        "PY\n"
    )


def _execute(ssh, command: str, *, timeout: int = 600, check: bool = True) -> tuple[str, str]:
    exit_code, stdout, stderr = ssh.execute_command(command, timeout=timeout)
    if check and exit_code != 0:
        raise RuntimeError(
            f"Remote command failed with exit {exit_code}: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stdout, stderr


def _uv_for_python(python_path: str) -> str:
    if python_path.endswith("/bin/python"):
        return python_path.rsplit("/", 1)[0] + "/uv"
    return "uv"


def open_session(pod_id: str, api_key: str, *, ssh_wait_timeout: int = 300, poll_interval: int = 5):
    deadline = time.monotonic() + ssh_wait_timeout
    ssh_details = None
    last_status = None
    while time.monotonic() < deadline:
        ssh_details = live_test_pkg.get_pod_ssh_details(pod_id, api_key)
        if ssh_details and ssh_details.get("ip") and ssh_details.get("port"):
            break
        try:
            last_status = live_test_pkg.get_pod_status(pod_id, api_key)
        except Exception:
            last_status = None
        time.sleep(poll_interval)
    if not ssh_details or not ssh_details.get("ip") or not ssh_details.get("port"):
        status_hint = _format_pod_status_hint(last_status)
        raise RuntimeError(
            f"Could not resolve SSH details for pod {pod_id} within {ssh_wait_timeout}s{status_hint}"
        )

    import os as _os
    private_key_path = _os.environ.get("REIGH_LIVE_TEST_SSH_KEY") or "~/.ssh/id_ed25519"
    ssh = live_test_pkg.SSHClient(
        hostname=str(ssh_details["ip"]),
        port=int(ssh_details["port"]),
        username="root",
        password=ssh_details.get("password"),
        private_key_path=private_key_path,
    )
    connect_deadline = time.monotonic() + ssh_wait_timeout
    last_err: Exception | None = None
    while time.monotonic() < connect_deadline:
        try:
            ssh.connect()
            return ssh
        except Exception as exc:
            last_err = exc
            try:
                last_status = live_test_pkg.get_pod_status(pod_id, api_key)
            except Exception:
                last_status = None
            time.sleep(poll_interval)
    status_hint = _format_pod_status_hint(last_status)
    raise RuntimeError(f"Could not SSH into pod {pod_id} within {ssh_wait_timeout}s{status_hint}: {last_err}")


def _format_pod_status_hint(status) -> str:
    if not isinstance(status, dict):
        return "; latest pod status unavailable"
    desired = status.get("desired_status") or status.get("desiredStatus") or "unknown"
    actual = status.get("actual_status") or status.get("actualStatus") or status.get("status") or "unknown"
    ip = status.get("ip") or "none"
    ports = status.get("ports")
    if isinstance(ports, list) and ports:
        rendered_ports = []
        for port in ports:
            if not isinstance(port, dict):
                continue
            private = port.get("privatePort") or port.get("private_port") or port.get("containerPort")
            public = port.get("publicPort") or port.get("public_port") or port.get("hostPort")
            if private and public:
                rendered_ports.append(f"{private}->{public}")
        ports_text = ",".join(rendered_ports) if rendered_ports else "unparseable"
    else:
        ports_text = "none"
    return f"; latest pod status desired={desired} actual={actual} ip={ip} ports={ports_text}"


def clone_repo_into(ssh, workdir: str, repo_url: str, branch: str) -> None:
    parent = workdir.rsplit("/", 1)[0] or "/"
    command = (
        f"mkdir -p {_quote(parent)} && "
        f"rm -rf {_quote(workdir)} && "
        f"git clone --branch {_quote(branch)} --single-branch --recurse-submodules {_quote(repo_url)} {_quote(workdir)}"
    )
    _execute(ssh, command, timeout=1800)


def _uv_sync_shell(
    workdir: str,
    *,
    env_path: str,
    extras: tuple[str, ...] = ("cuda124",),
    with_locked: bool = False,
) -> str:
    """Return the shell script body that runs ``uv sync`` with retry semantics.

    Includes the ``cd``, PATH/env exports, and the retry loop. Callers must
    add any preamble (apt-get install, uv install) themselves.
    """
    if not extras:
        raise ValueError("_uv_sync_shell requires a non-empty extras tuple")
    extras_args = " ".join(f"--extra {extra}" for extra in extras)
    locked_flag = " --locked" if with_locked else ""
    return (
        f"cd {shlex.quote(workdir)}\n"
        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
        f"export UV_PROJECT_ENVIRONMENT={env_path}\n"
        "export UV_LINK_MODE=copy\n"
        "for attempt in 1 2 3; do\n"
        f"  if uv sync{locked_flag} {extras_args}; then\n"
        "    break\n"
        "  fi\n"
        "  echo \"uv sync attempt $attempt failed; cleaning partial venv and retrying\"\n"
        "  rm -rf .venv \"$UV_PROJECT_ENVIRONMENT\"\n"
        "  sleep 5\n"
        "  if [ $attempt -eq 3 ]; then exit 1; fi\n"
        "done\n"
    )


def _vibecomfy_install_shell(
    workdir: str,
    *,
    python_path: str,
    attention_profile: str | None,
    run_nodes_restore: bool = True,
) -> str:
    """Return the shell script body that installs VibeComfy + ComfyUI into ``workdir``.

    Assumes the repo has already been cloned/checked-out at ``workdir``. The
    caller may compose this snippet behind a fresh clone (cold install) or
    behind a ``git fetch && reset --hard`` (warm sync).
    """
    resolved = _resolve_attention_profile(attention_profile)
    if "/" in python_path:
        install_python = python_path
        uv_path = _uv_for_python(python_path)
        venv_block = ""
    else:
        install_python = f"{workdir}/.venv/bin/python"
        uv_path = "uv"
        venv_block = f"uv venv --python {_quote(python_path)} {_quote(workdir)}/.venv\n"

    uv = _quote(uv_path)
    install_py = _quote(install_python)
    sage = _sageattention_install_block(install_python, uv_path=uv_path) if resolved == "sage" else ""
    nodes_block = ""
    if run_nodes_restore:
        nodes_block = (
            f"cd {_quote(workdir)}\n"
            "test -f custom_nodes.lock\n"
            f"{install_py} -m vibecomfy.cli nodes restore --lockfile custom_nodes.lock\n"
            f"test -f {_quote(workdir)}/template_index.json\n"
            f"test -f {_quote(workdir)}/workflow_corpus/manifests/coverage.json\n"
        )
    return (
        f"{venv_block}"
        f"{uv} pip install --python {install_py} -e {_quote(workdir)}\n"
        f"{uv} pip install --python {install_py} "
        "'comfyui@git+https://github.com/peteromallet/ComfyUI.git@fix/latentupscale-model-mmap-residency' "
        "'comfy-script[default]'\n"
        f"{sage}"
        f"{nodes_block}"
    )


def run_install(ssh, workdir: str) -> None:
    package_list = " ".join(APT_INSTALL_PACKAGES)
    sync_shell = _uv_sync_shell(
        workdir,
        env_path="/opt/reigh-worker-live-test-venv",
        extras=("cuda124",),
        with_locked=False,
    )
    command = (
        "bash -lc "
        + _quote(
            "set -euo pipefail\n"
            "apt-get update\n"
            f"apt-get install -y {package_list}\n"
            "if ! command -v uv >/dev/null 2>&1; then\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  export PATH=\"$HOME/.local/bin:$PATH\"\n"
            "fi\n"
            + sync_shell
        )
    )
    _execute(ssh, command, timeout=3600)


def clone_and_install_vibecomfy(
    ssh,
    *,
    repo_url: str,
    branch: str,
    workdir: str = "/workspace/vibecomfy",
    python_path: str,
    attention_profile: str | None = None,
) -> None:
    parent = workdir.rsplit("/", 1)[0] or "/"
    resolved_attention_profile = _resolve_attention_profile(attention_profile)
    install_shell = _vibecomfy_install_shell(
        workdir,
        python_path=python_path,
        attention_profile=attention_profile,
        run_nodes_restore=True,
    )
    command = (
        "bash -lc "
        + _quote(
            "set -euo pipefail\n"
            f"export VIBECOMFY_ATTENTION_PROFILE={_quote(resolved_attention_profile)}\n"
            f"mkdir -p {_quote(parent)}\n"
            f"rm -rf {_quote(workdir)}\n"
            f"git clone --branch {_quote(branch)} --single-branch {_quote(repo_url)} {_quote(workdir)}\n"
            f"git -C {_quote(workdir)} fetch origin {_quote(branch)}\n"
            f"git -C {_quote(workdir)} checkout {_quote(branch)}\n"
            f"git -C {_quote(workdir)} reset --hard FETCH_HEAD\n"
            f"git -C {_quote(workdir)} clean -ffd -e .venv/\n"
            f"echo \"VibeComfy checkout: $(git -C {_quote(workdir)} rev-parse --short HEAD)\"\n"
            + install_shell
        )
    )
    _execute(ssh, command, timeout=3600)


def bundle_venv(ssh, *, source_env_path: str, bundle_path: str) -> str:
    """Tar+zstd the venv at *source_env_path* into *bundle_path*; return SHA256."""
    return _bundle_directory(ssh, source_path=source_env_path, bundle_path=bundle_path)


def bundle_install_tree(ssh, *, source_path: str, bundle_path: str) -> str:
    """Tar+zstd an install tree (e.g. VibeComfy checkout) at *source_path*; return SHA256."""
    return _bundle_directory(ssh, source_path=source_path, bundle_path=bundle_path)


def _bundle_directory(ssh, *, source_path: str, bundle_path: str) -> str:
    parent_dir = bundle_path.rsplit("/", 1)[0] or "/"
    staging = f"{bundle_path}.staging"
    source_parent = source_path.rsplit("/", 1)[0] or "/"
    source_name = source_path.rsplit("/", 1)[1] if "/" in source_path else source_path
    script = (
        "set -euo pipefail\n"
        f"mkdir -p {_quote(parent_dir)}\n"
        f"rm -f {_quote(staging)}\n"
        f"tar --use-compress-program 'zstd -1 --threads=0' "
        f"-cf {_quote(staging)} -C {_quote(source_parent)} {_quote(source_name)}\n"
        f"sha256sum {_quote(staging)} | awk '{{print $1}}'\n"
        f"mv {_quote(staging)} {_quote(bundle_path)}\n"
    )
    stdout, _stderr = _execute(ssh, "bash -lc " + _quote(script), timeout=7200)
    digest = ""
    for line in stdout.splitlines():
        candidate = line.strip()
        if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate.lower()):
            digest = candidate.lower()
    if not digest:
        raise RuntimeError(
            f"bundle_directory failed to capture sha256sum output for {source_path} -> {bundle_path}; stdout={stdout!r}"
        )
    return digest


def extract_bundle_to_container_disk(
    ssh,
    *,
    bundle_path: str,
    target_path: str,
    expected_sha256: str,
) -> None:
    """Verify the SHA256 of *bundle_path* matches *expected_sha256*, then extract it into *target_path*."""
    target_parent = target_path.rsplit("/", 1)[0] or "/"
    sha_script = (
        "set -euo pipefail\n"
        f"sha256sum {_quote(bundle_path)} | awk '{{print $1}}'\n"
    )
    stdout, _stderr = _execute(ssh, "bash -lc " + _quote(sha_script), timeout=600)
    observed = stdout.strip().splitlines()[-1].strip().lower() if stdout.strip() else ""
    if observed != expected_sha256.lower():
        raise RuntimeError(
            "extract_bundle_to_container_disk sha256 mismatch for "
            f"{bundle_path}: expected {expected_sha256}, observed {observed!r}"
        )
    extract_script = (
        "set -euo pipefail\n"
        f"mkdir -p {_quote(target_parent)}\n"
        f"rm -rf {_quote(target_path)}\n"
        f"mkdir -p {_quote(target_path)}\n"
        f"if command -v pv >/dev/null 2>&1; then\n"
        f"  pv {_quote(bundle_path)} | tar --use-compress-program 'zstd -d --threads=0' -xf - -C {_quote(target_path)} --strip-components=1\n"
        f"else\n"
        f"  tar --use-compress-program 'zstd -d --threads=0' -xf {_quote(bundle_path)} -C {_quote(target_path)} --strip-components=1\n"
        f"fi\n"
    )
    exit_code, ex_stdout, ex_stderr = ssh.execute_command(
        "bash -lc " + _quote(extract_script), timeout=3600
    )
    if exit_code != 0:
        err_lines = (ex_stderr or "").splitlines()
        first = err_lines[:50]
        last = err_lines[-50:] if len(err_lines) > 50 else []
        diag = "\n".join(first + (["..."] + last if last else []))
        raise RuntimeError(
            f"extract_bundle_to_container_disk failed extracting {bundle_path} -> {target_path}; "
            f"exit={exit_code}; stderr:\n{diag}\nstdout:\n{ex_stdout}"
        )


def ensure_git_ref_synced(
    ssh,
    *,
    workdir: str,
    repo_url: str,
    ref: str,
    force_clone: bool = False,
) -> None:
    """Materialize *ref* of *repo_url* at *workdir* without running any uv sync."""
    parent = workdir.rsplit("/", 1)[0] or "/"
    if force_clone:
        script = (
            "set -euo pipefail\n"
            f"mkdir -p {_quote(parent)}\n"
            f"rm -rf {_quote(workdir)}\n"
            f"git clone {_quote(repo_url)} {_quote(workdir)}\n"
            f"git -C {_quote(workdir)} fetch origin {_quote(ref)}\n"
            f"git -C {_quote(workdir)} checkout {_quote(ref)}\n"
            f"git -C {_quote(workdir)} reset --hard FETCH_HEAD\n"
            f"git -C {_quote(workdir)} clean -ffd -e .venv/\n"
        )
    else:
        script = (
            "set -euo pipefail\n"
            f"git -C {_quote(workdir)} fetch origin {_quote(ref)}\n"
            f"git -C {_quote(workdir)} checkout {_quote(ref)}\n"
            f"git -C {_quote(workdir)} reset --hard FETCH_HEAD\n"
            f"git -C {_quote(workdir)} clean -ffd -e .venv/\n"
        )
    _execute(ssh, "bash -lc " + _quote(script), timeout=1800)


def export_env(env: dict[str, str]) -> str:
    exports = dict(env)
    required = {
        "REIGH_ACCESS_TOKEN",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_URL",
        "WORKER_DB_CLIENT_AUTH_MODE",
    }
    missing = sorted(name for name in required if not exports.get(name))
    if missing:
        raise ValueError(f"Missing required environment values for export_env: {', '.join(missing)}")
    if exports["WORKER_DB_CLIENT_AUTH_MODE"] not in {"worker", "service"}:
        raise ValueError("WORKER_DB_CLIENT_AUTH_MODE must be 'worker' or 'service' for live tests")
    return " && ".join(f"export {key}={_quote(value)}" for key, value in sorted(exports.items()))


def capture_current_worker_cmdline(ssh) -> WorkerProcessInfo | None:
    stdout, _ = _execute(ssh, PROCESS_SCAN_COMMAND, check=False)
    rows: list[tuple[int, list[str]]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, args = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        rows.append((pid, shlex.split(args)))

    supervisor_rows = [row for row in rows if any("run_worker.py" in arg for arg in row[1])]
    if supervisor_rows:
        pid, cmdline = supervisor_rows[0]
        return WorkerProcessInfo(family="supervisor", cmdline=cmdline, pid=pid)

    direct_rows = [
        row
        for row in rows
        if any(arg == "worker.py" or arg.endswith("/worker.py") or "source.runtime.worker" in arg for arg in row[1])
    ]
    if direct_rows:
        pid, cmdline = direct_rows[0]
        return WorkerProcessInfo(family="direct", cmdline=cmdline, pid=pid)

    return None


def kill_supervisor_and_worker(ssh) -> None:
    _execute(ssh, KILL_COMMAND, check=False, timeout=30)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        stdout, _ = _execute(
            ssh,
            PROCESS_SCAN_COMMAND,
            check=False,
            timeout=30,
        )
        if not stdout.strip():
            return
        time.sleep(1)
    raise RuntimeError(f"Worker processes are still running after kill attempt:\n{stdout}")


def launch_worker_detached(ssh, command_line: str) -> None:
    # Wrap so bash exits immediately after backgrounding; avoid paramiko hanging
    # on stdout EOF when a nohup child inherits the ssh channel streams.
    wrapped = f"bash -c {shlex.quote(command_line + ' ; exit 0')} </dev/null >/dev/null 2>&1"
    client = ssh.client
    transport = client.get_transport()
    channel = transport.open_session()
    try:
        channel.set_combine_stderr(True)
        channel.exec_command(wrapped)
        deadline = time.monotonic() + 30
        while not channel.exit_status_ready() and time.monotonic() < deadline:
            time.sleep(0.2)
        if not channel.exit_status_ready():
            return  # detached; don't wait further
        exit_code = channel.recv_exit_status()
        if exit_code != 0:
            raise RuntimeError(f"Detached launch command exited with {exit_code}")
    finally:
        channel.close()


def fetch_worker_logs(ssh, workdir: str, lines: int = 300) -> str:
    startup_script = _quote(
        f"cd {workdir} && "
        f'{{ echo "=== startup.log ==="; tail -n {int(lines)} logs/startup.log 2>/dev/null || true; }}'
    )
    startup_stdout, _ = _execute(
        ssh,
        f"bash -lc {startup_script}",
        check=False,
        timeout=60,
    )
    worker_script = _quote(
        f"cd {workdir} && "
        f'{{ echo "=== worker.log ==="; tail -n {int(lines)} logs/worker.log 2>/dev/null || true; }}'
    )
    worker_stdout, _ = _execute(
        ssh,
        f"bash -lc {worker_script}",
        check=False,
        timeout=60,
    )
    return "\n".join(part.rstrip() for part in (startup_stdout, worker_stdout) if part.strip())


__all__ = [
    "APT_INSTALL_PACKAGES",
    "KILL_COMMAND",
    "PROCESS_SCAN_COMMAND",
    "WorkerProcessInfo",
    "bundle_install_tree",
    "bundle_venv",
    "capture_current_worker_cmdline",
    "clone_and_install_vibecomfy",
    "clone_repo_into",
    "ensure_git_ref_synced",
    "export_env",
    "extract_bundle_to_container_disk",
    "fetch_worker_logs",
    "kill_supervisor_and_worker",
    "launch_worker_detached",
    "open_session",
    "run_install",
]
