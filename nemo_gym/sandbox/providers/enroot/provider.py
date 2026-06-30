# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enroot sandbox provider.

Enroot is NVIDIA's unprivileged container runtime (https://github.com/NVIDIA/enroot).
This provider is a peer of the apptainer/docker providers and is deliberately
runtime-only / scheduler-agnostic: it shells out to the ``enroot`` CLI and reads
``ENROOT_*`` paths from the environment. It makes no assumptions about Slurm,
pyxis, or any site layout -- callers pass images, binds, and env via the spec/config.

Model vs. apptainer:
  * apptainer keeps a daemonized ``instance``; enroot has no daemon. ``create``
    unpacks a writable rootfs (``enroot create``) and each ``exec`` is one
    ``enroot start --rw`` against that container, so the rootfs (e.g. ``/testbed``
    edits) persists across execs. Binds/env are therefore applied on every start.
  * enroot's rootfs is natively writable -- no read-only-image overlay needed.
  * enroot does not bind the host ``$HOME`` -- no ``--no-mount home`` needed.
"""
import asyncio
import contextlib
import logging
import os
import shlex
import shutil
import signal
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)

LOGGER = logging.getLogger(__name__)

CONTAINER_NAME_PREFIX = "nemo-gym-enroot-"
# Sentinel return code when enroot itself failed to run the command (no process exit code).
SANDBOX_RUNTIME_RETURN_CODE = -1


class EnrootCreateError(SandboxCreateError):
    """Raised when the enroot provider cannot create a sandbox."""


class EnrootCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a newly-created enroot sandbox cannot execute a probe command."""


def _require_enroot() -> str:
    path = shutil.which("enroot")
    if not path:
        raise RuntimeError(
            "The 'enroot' binary is required for the enroot sandbox provider but was not found on PATH."
        )
    return path


def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
    """Accept either a config dataclass instance or a plain mapping (Hydra YAML)."""
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**dict(value))
    raise TypeError(f"Expected {config_cls.__name__} or mapping, got {type(value).__name__}")


def _coerce_binds(value: Any) -> list[str]:
    """Accept a list of 'src:dst[:ro|rw]' strings (or a single such string)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(b) for b in value]
    raise TypeError(f"binds must be a string or list of strings, got {type(value).__name__}")


def _to_enroot_mount(bind: str) -> str:
    """Translate a neutral 'src:dst[:ro|rw]' bind into enroot --mount fstab syntax.

    enroot expects ``src:dst:type:options``. ``x-create=dir`` makes the mount
    point if it does not exist; ``bind`` + rw/ro gives a bind mount.
    """
    parts = bind.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid bind {bind!r}; expected 'src:dst[:ro|rw]'")
    src, dst = parts[0], parts[1]
    mode = parts[2] if len(parts) > 2 and parts[2] else "rw"
    mode = "ro" if mode == "ro" else "rw"
    return f"{src}:{dst}:none:x-create=dir,bind,{mode}"


@dataclass
class EnrootCreateConfig:
    # Timeout for `enroot import`/`enroot create` (image unpack can be multi-GB / slow).
    start_timeout_s: float = 1800.0
    # Extra flags appended to `enroot create`.
    extra_create_args: list[str] = field(default_factory=list)
    # Where docker:// imports are cached as .sqsh. None -> $ENROOT_CACHE_PATH or a temp dir.
    image_cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.start_timeout_s <= 0:
            raise ValueError("create.start_timeout_s must be > 0")


@dataclass
class EnrootExecConfig:
    default_timeout_s: float = 3600.0
    concurrency: int = 8
    # Neutral 'src:dst[:ro|rw]' binds applied to every exec/start in this sandbox.
    default_binds: list[str] = field(default_factory=list)
    # Extra flags appended to every `enroot start`.
    extra_start_args: list[str] = field(default_factory=list)
    rw: bool = True  # writable rootfs so /testbed edits persist across execs
    root: bool = True  # run as uid 0 in the container (SWE tasks edit/build under root)

    def __post_init__(self) -> None:
        if self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")


@dataclass
class EnrootProbeConfig:
    command: str | None = "true"
    timeout_s: float = 30.0
    deadline_s: float | None = 120.0
    stable_count: int = 1
    expect_returncode: int = 0

    def __post_init__(self) -> None:
        if self.stable_count < 1:
            raise ValueError("probe.stable_count must be >= 1")


@dataclass
class _EnrootContainer:
    """Provider-private state stored on SandboxHandle.raw."""

    name: str
    sqsh: str
    binds: list[str]  # neutral 'src:dst[:ro|rw]'
    env: dict[str, str]


def _to_sandbox_status(present: bool) -> SandboxStatus:
    # enroot has no run daemon; a created container is "ready to exec".
    return SandboxStatus.RUNNING if present else SandboxStatus.STOPPED


class EnrootProvider:
    """SandboxProvider backed by the ``enroot`` CLI."""

    name = "enroot"

    def __init__(
        self,
        *,
        exec: EnrootExecConfig | Mapping[str, Any] | None = None,
        create: EnrootCreateConfig | Mapping[str, Any] | None = None,
        probe: EnrootProbeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._exec_config = _coerce_config(exec, EnrootExecConfig)
        self._create_config = _coerce_config(create, EnrootCreateConfig)
        self._probe = _coerce_config(probe, EnrootProbeConfig)
        self._binary = _require_enroot()
        self._semaphore = asyncio.Semaphore(self._exec_config.concurrency)
        self._import_lock = asyncio.Lock()

    async def _run(
        self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None
    ) -> tuple[int, str, str]:
        """Run an enroot CLI command. Returns (return_code, stdout, stderr).

        Kills the whole process group on timeout so children do not linger;
        bounds concurrency with a shared semaphore; decodes with errors='replace'.
        """
        async with self._semaphore:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(input=stdin), timeout=timeout_s)
            except asyncio.TimeoutError as e:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await proc.wait()
                raise TimeoutError(f"enroot command timed out after {timeout_s:g}s: {argv}") from e
            return_code = proc.returncode if proc.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
            return return_code, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")

    def _cache_dir(self) -> Path:
        d = (
            self._create_config.image_cache_dir
            or os.environ.get("ENROOT_CACHE_PATH")
            or os.path.join(tempfile.gettempdir(), "nemo-gym-enroot-cache")
        )
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _resolve_sqsh(self, image: str) -> str:
        """Return a local .sqsh path for ``image``, importing from docker:// on first use.

        A local ``.sqsh`` path is used verbatim. Anything else is treated as a
        docker reference and imported (once, serialized) into the image cache.
        """
        if image.endswith(".sqsh") and os.path.exists(image):
            return image
        ref = image[len("docker://") :] if image.startswith("docker://") else image
        safe = ref.replace("/", "+").replace(":", "+")
        sqsh = self._cache_dir() / f"{safe}.sqsh"
        if sqsh.exists():
            return str(sqsh)
        async with self._import_lock:
            if sqsh.exists():  # another task imported it while we waited
                return str(sqsh)
            tmp = sqsh.with_name(sqsh.name + ".tmp")
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            code, _out, err = await self._run(
                [self._binary, "import", "-o", str(tmp), f"docker://{ref}"],
                timeout_s=self._create_config.start_timeout_s,
            )
            if code != 0:
                with contextlib.suppress(FileNotFoundError):
                    tmp.unlink()
                raise EnrootCreateError(f"enroot import failed for docker://{ref} (code={code}): {err.strip()}")
            os.replace(tmp, sqsh)
        return str(sqsh)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.ttl_s is not None:
            LOGGER.warning("ttl_s is not supported by the enroot provider; it will be ignored.")
        if spec.image is None:
            raise EnrootCreateError("spec.image is required for the enroot provider")

        sqsh = await self._resolve_sqsh(spec.image)
        name = CONTAINER_NAME_PREFIX + uuid.uuid4().hex

        argv = [self._binary, "create", "-n", name, *self._create_config.extra_create_args, sqsh]
        try:
            code, _out, err = await self._run(argv, timeout_s=self._create_config.start_timeout_s)
        except TimeoutError as e:
            raise EnrootCreateError(f"enroot create timed out for image={spec.image!r}: {e}") from e
        if code != 0:
            raise EnrootCreateError(f"enroot create failed (code={code}) for image={spec.image!r}: {err.strip()}")

        binds = list(self._exec_config.default_binds) + _coerce_binds(spec.provider_options.get("binds"))
        handle = SandboxHandle(
            sandbox_id=name,
            provider_name=self.name,
            raw=_EnrootContainer(name=name, sqsh=sqsh, binds=binds, env=dict(spec.env)),
        )
        try:
            await self._verify_created_handle(handle)
        except Exception:
            await self._cleanup(handle)
            raise
        return handle

    def _start_argv(
        self, inst: _EnrootContainer, command: str, *, env_extra: dict[str, str] | None, cwd: str | None,
        extra_binds: list[str] | None = None,
    ) -> list[str]:
        argv = [self._binary, "start"]
        if self._exec_config.root:
            argv.append("--root")
        if self._exec_config.rw:
            argv.append("--rw")
        for bind in inst.binds + (extra_binds or []):
            argv += ["--mount", _to_enroot_mount(bind)]
        merged_env = {**inst.env, **(env_extra or {})}
        for key, value in merged_env.items():
            argv += ["-e", f"{key}={value}"]
        argv += self._exec_config.extra_start_args
        argv.append(inst.name)
        inner = command if not cwd else f"cd {shlex.quote(cwd)} && {command}"
        # Use 'sh' (universally present, incl. minimal images) rather than 'bash', matching the
        # apptainer provider; enroot's /etc/rc hook execs this, so the interpreter must exist.
        argv += ["sh", "-c", inner]
        return argv

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        inst: _EnrootContainer = handle.raw
        argv = self._start_argv(inst, command, env_extra=env, cwd=cwd)
        effective_timeout = timeout_s if timeout_s is not None else self._exec_config.default_timeout_s
        try:
            code, out, err = await self._run(argv, timeout_s=effective_timeout)
        except TimeoutError:
            return SandboxExecResult(
                stdout=None,
                stderr=f"enroot exec timed out after {effective_timeout:g}s",
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="timeout",
            )
        return SandboxExecResult(stdout=out, stderr=err, return_code=code)

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        probe = self._probe
        if probe.command is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + probe.deadline_s if probe.deadline_s is not None else None
        consecutive = 0
        last_detail = "no probe attempt completed"
        while True:
            result = await self.exec(handle, probe.command, timeout_s=probe.timeout_s)
            if result.return_code == probe.expect_returncode:
                consecutive += 1
                if consecutive >= probe.stable_count:
                    return
            else:
                consecutive = 0
                last_detail = (result.stderr or result.stdout or "").strip()[:300] or f"code={result.return_code}"
            if deadline is None:
                raise EnrootCreateVerificationError(f"enroot sandbox probe failed: {last_detail}")
            if loop.time() >= deadline:
                raise EnrootCreateVerificationError(f"enroot sandbox not ready before deadline: {last_detail}")
            await asyncio.sleep(min(2.0, probe.timeout_s))

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        inst: _EnrootContainer = handle.raw
        src = Path(source_path).resolve()
        mnt = "/__nemo_gym_upload"
        cmd = f"cp {shlex.quote(f'{mnt}/{src.name}')} {shlex.quote(target_path)}"
        argv = self._start_argv(inst, cmd, env_extra=None, cwd=None, extra_binds=[f"{src.parent}:{mnt}:ro"])
        code, _out, err = await self._run(argv, timeout_s=self._exec_config.default_timeout_s)
        if code != 0:
            raise RuntimeError(f"enroot upload_file failed (code={code}): {err.strip()}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        inst: _EnrootContainer = handle.raw
        tgt = Path(target_path).resolve()
        tgt.parent.mkdir(parents=True, exist_ok=True)
        mnt = "/__nemo_gym_download"
        cmd = f"cp {shlex.quote(source_path)} {shlex.quote(f'{mnt}/{tgt.name}')}"
        argv = self._start_argv(inst, cmd, env_extra=None, cwd=None, extra_binds=[f"{tgt.parent}:{mnt}:rw"])
        code, _out, err = await self._run(argv, timeout_s=self._exec_config.default_timeout_s)
        if code != 0:
            raise RuntimeError(f"enroot download_file failed (code={code}): {err.strip()}")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        code, out, _err = await self._run([self._binary, "list"], timeout_s=60.0)
        if code != 0:
            return SandboxStatus.UNKNOWN
        present = any(line.strip() == handle.sandbox_id for line in out.splitlines())
        return _to_sandbox_status(present)

    async def close(self, handle: SandboxHandle) -> None:
        await self._cleanup(handle)

    async def _cleanup(self, handle: SandboxHandle) -> None:
        with contextlib.suppress(Exception):
            await self._run([self._binary, "remove", "-f", handle.sandbox_id], timeout_s=120.0)

    async def aclose(self) -> None:
        # No provider-scoped resources (no SDK client / daemon) to release.
        return None
