"""BoxLite microVM execution environment backend for code execution.

BoxLite (https://github.com/boxlite-ai/boxlite) runs OCI images inside
KVM-backed microVMs. This backend talks to a *remote* ``boxlite serve`` REST
server (configured via ``BOXLITE_REST_URL`` or the ``url`` argument), so a
single host with hardware virtualization can serve isolated sandboxes to many
eval workers — the same self-hosting model as a private E2B, but on your own
machine.

Structurally this mirrors :class:`E2BCodeExecToolProvider`: a box is created in
``__aenter__``, commands run via ``exec``, files move with ``copy_in`` /
``copy_out``, and the box is removed in ``__aexit__``.

Two BoxLite-specific details shape the implementation:

1. **Working directory.** The default working dir is ``/home/user`` (matching
   E2B), but a stock image such as ``python:3.12-slim`` does not contain it.
   Setting ``BoxOptions.working_dir`` to a path that does not exist makes the
   microVM fail to spawn, so we create the box *without* a working dir, then
   ``mkdir -p`` it on the first (spawn-triggering) ``exec`` and pass ``cwd``
   explicitly on every command thereafter.

2. **``copy_in`` wraps content in an ``extracted/`` subdirectory.** A
   ``copy_in(host, dest)`` lands the payload at ``dest/extracted/<name>`` rather
   than ``dest/<name>``. We therefore upload via a single tarball (preserving
   directory structure) and untar it into place, and stage single-file writes
   through a unique scratch dir before moving the file to its final path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import shlex
import tarfile
import tempfile
from contextlib import AbstractAsyncContextManager
from pathlib import Path

try:
    from boxlite import (
        ApiKeyCredential,
        BoxOptions,
        Boxlite,
        BoxliteRestOptions,
    )
except ImportError as e:
    raise ImportError(
        "Requires installation of the boxlite extra. Install with (for example): "
        "`uv pip install stirrup[boxlite]` or `uv add stirrup[boxlite]`",
    ) from e

from stirrup.core.models import ImageContentBlock, Tool, ToolUseCountMetadata

from .base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
    UploadedFile,
    UploadFilesResult,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKING_DIR = "/home/user"
DEFAULT_IMAGE = "python:3.12-slim"


class BoxliteCodeExecToolProvider(CodeExecToolProvider):
    """BoxLite microVM code execution tool provider.

    Usage with Agent:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient
        from stirrup.tools.code_backends.boxlite import BoxliteCodeExecToolProvider

        provider = BoxliteCodeExecToolProvider(
            url="http://localhost:8100",            # a `boxlite serve` endpoint
            image="python:3.12-slim",
            setup_commands=["pip install -q numpy pandas"],
        )
        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(client=client, name="assistant", tools=[provider])
        async with agent.session(output_dir="./out", input_files="data/") as session:
            await session.run("Analyze the data")

    Standalone usage:
        provider = BoxliteCodeExecToolProvider(url="http://localhost:8100")
        async with provider as tool:
            result = await provider.run_command("python --version")
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        prefix: str | None = None,
        image: str = DEFAULT_IMAGE,
        working_dir: str = DEFAULT_WORKING_DIR,
        disk_size_gb: int | None = 8,
        cpus: int | None = 2,
        memory_mib: int | None = 2048,
        setup_commands: list[str] | None = None,
        setup_timeout: int = 600,
        allowed_commands: list[str] | None = None,
        create_gate: AbstractAsyncContextManager[object] | None = None,
        box_options_kwargs: dict | None = None,
        shell_timeout: int = SHELL_TIMEOUT,
    ) -> None:
        """Initialize BoxLite execution environment configuration.

        Args:
            url: BoxLite REST server URL (``boxlite serve`` endpoint, e.g.
                ``http://localhost:8100``). Defaults to ``BOXLITE_REST_URL``.
                If neither is set, the in-process local runtime is used
                (requires this process to own ``$BOXLITE_HOME``).
            api_key: Optional API key for the server. Defaults to
                ``BOXLITE_API_KEY``. ``None`` means no credential (fine for a
                localhost / SSH-tunneled server).
            prefix: Optional REST path prefix forwarded to ``BoxliteRestOptions``.
            image: OCI image to boot the microVM from (default: ``python:3.12-slim``).
            working_dir: Working directory for commands and the default upload
                destination (default: ``/home/user``). Created on box start.
            disk_size_gb: Root disk size in GB. The default COW overlay is tiny
                (~256 MB) — too small to ``pip install`` numpy/pandas — so this
                defaults to 8. ``None`` uses the image-derived default.
            cpus: vCPUs for the microVM. ``None`` uses the server default.
            memory_mib: Memory (MiB) for the microVM. ``None`` uses the default.
            setup_commands: Shell commands run once at box start (after the
                working dir is created), e.g. to install Python packages. Joined
                with ``&&``; a non-zero exit aborts startup.
            setup_timeout: Per-call timeout (seconds) for the startup command.
            allowed_commands: Optional regex allowlist; only matching commands run.
            create_gate: Optional async context manager entered immediately
                before each box creation — use to throttle/serialize creation
                when many providers start concurrently.
            box_options_kwargs: Extra keyword arguments merged into ``BoxOptions``
                (e.g. ``user``, ``entrypoint``). Explicit args above take
                precedence over matching keys here.
            shell_timeout: Per-command wall-clock timeout (seconds) for every
                ``code_exec`` invocation and the default for direct
                ``run_command`` calls passing ``timeout=None``.
        """
        super().__init__(allowed_commands=allowed_commands, shell_timeout=shell_timeout)
        self._url = url if url is not None else os.environ.get("BOXLITE_REST_URL")
        self._api_key = api_key if api_key is not None else os.environ.get("BOXLITE_API_KEY")
        self._prefix = prefix
        self._image = image
        self._working_dir = working_dir.rstrip("/") or "/"
        self._disk_size_gb = disk_size_gb
        self._cpus = cpus
        self._memory_mib = memory_mib
        self._setup_commands = setup_commands or []
        self._setup_timeout = setup_timeout
        self._create_gate = create_gate
        self._box_options_kwargs = box_options_kwargs or {}

        # Runtime state
        self._rt: Boxlite | None = None
        self._box = None

    @property
    def box_id(self) -> str | None:
        """Return the box id, or None if not started."""
        return self._box.id if self._box else None

    # -- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        """Create the microVM, prepare the working dir + setup, return the tool."""
        if self._url:
            credential = ApiKeyCredential(self._api_key) if self._api_key else None
            self._rt = Boxlite.rest(BoxliteRestOptions(url=self._url, credential=credential, prefix=self._prefix))
        else:
            # In-process runtime. Only one process may own $BOXLITE_HOME at a time.
            self._rt = Boxlite.default()

        opts_kwargs: dict = {"image": self._image, **self._box_options_kwargs}
        # NOTE: deliberately NOT setting working_dir here — a non-existent dir in
        # the image makes the microVM fail to spawn. We mkdir it below.
        if self._disk_size_gb is not None:
            opts_kwargs["disk_size_gb"] = self._disk_size_gb
        if self._cpus is not None:
            opts_kwargs["cpus"] = self._cpus
        if self._memory_mib is not None:
            opts_kwargs["memory_mib"] = self._memory_mib

        if self._create_gate is not None:
            async with self._create_gate:
                self._box = await self._rt.create(BoxOptions(**opts_kwargs))
        else:
            self._box = await self._rt.create(BoxOptions(**opts_kwargs))

        # First exec triggers the actual VM spawn; run it with cwd=None (the
        # working dir does not exist yet) and create the working dir + run setup.
        startup = " && ".join([f"mkdir -p {shlex.quote(self._working_dir)}", *self._setup_commands])
        result = await self._raw_exec(startup, cwd=None, timeout=self._setup_timeout)
        if result.exit_code != 0:
            box_id = self._box.id if self._box else "?"
            await self._cleanup_box()
            raise RuntimeError(
                f"BoxLite box {box_id} startup failed (exit {result.exit_code}): "
                f"{result.stderr[-2000:] or result.stdout[-2000:]}"
            )

        logger.info(
            "Started BoxLite box %s (image=%s, workdir=%s)", self._box.id, self._image, self._working_dir
        )
        return self.get_code_exec_tool()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Remove the microVM."""
        await self._cleanup_box()

    async def _cleanup_box(self) -> None:
        if self._box is not None and self._rt is not None:
            box_id = self._box.id
            try:
                await self._rt.remove(box_id)
                logger.info("Removed BoxLite box %s", box_id)
            except Exception as exc:  # noqa: BLE001 - cleanup must not raise
                logger.warning("Failed to remove BoxLite box %s: %s", box_id, exc)
        self._box = None
        self._rt = None

    # -- exec ---------------------------------------------------------------

    class _Raw:
        __slots__ = ("exit_code", "stdout", "stderr", "error_message")

        def __init__(self, exit_code: int, stdout: str, stderr: str, error_message: str | None) -> None:
            self.exit_code = exit_code
            self.stdout = stdout
            self.stderr = stderr
            self.error_message = error_message

    async def _raw_exec(self, cmd: str, *, cwd: str | None, timeout: int | None) -> "BoxliteCodeExecToolProvider._Raw":
        """Run ``bash -lc cmd`` in the box; collect stdout/stderr; return raw result.

        Streams must each be taken exactly once and drained concurrently (a
        sequential read can deadlock when one pipe fills while we block on the
        other), then ``wait()`` yields the exit code.
        """
        if self._box is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure the current Agent is equipped with a CodeExecToolProvider."
            )
        execution = await self._box.exec("bash", args=["-lc", cmd], cwd=cwd, timeout_secs=timeout)

        out: list[str] = []
        err: list[str] = []

        async def drain(stream, acc: list[str]) -> None:
            async for line in stream:
                acc.append(line.decode("utf-8", errors="replace") if isinstance(line, (bytes, bytearray)) else line)

        await asyncio.gather(drain(execution.stdout(), out), drain(execution.stderr(), err))
        result = await execution.wait()
        return self._Raw(getattr(result, "exit_code", 0), "".join(out), "".join(err), getattr(result, "error_message", None))

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        """Execute a shell command in the box working directory.

        Args:
            cmd: Shell command to execute (bash syntax).
            timeout: Per-call wall-clock timeout (seconds). If None, falls back
                to ``self._shell_timeout`` configured on the provider.
        """
        if timeout is None:
            timeout = self._shell_timeout

        if not self._check_allowed(cmd):
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: '{cmd}' does not match any allowed patterns",
                error_kind="command_not_allowed",
                advice="Only commands matching the allowlist patterns are permitted.",
            )

        try:
            # BoxLite enforces timeout_secs inside the guest; the outer wait_for
            # is a safety net for a stalled transport.
            raw = await asyncio.wait_for(
                self._raw_exec(cmd, cwd=self._working_dir, timeout=timeout),
                timeout=timeout + 15,
            )
        except asyncio.TimeoutError:
            logger.warning("Command timed out after %d seconds: %s", timeout, cmd[:100])
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )
        except Exception as exc:  # noqa: BLE001 - surface as a tool error, not a crash
            return CommandResult(exit_code=1, stdout="", stderr=str(exc), error_kind="execution_error")

        # A guest-side timeout(1)/SIGKILL surfaces as 124/137 like the Docker backend.
        if raw.exit_code in (124, 137):
            return CommandResult(
                exit_code=raw.exit_code,
                stdout=raw.stdout,
                stderr=(raw.stderr + f"\nCommand timed out after {timeout} seconds").strip(),
                error_kind="timeout",
            )
        return CommandResult(exit_code=raw.exit_code, stdout=raw.stdout, stderr=raw.stderr)

    # -- path helpers -------------------------------------------------------

    def _resolve(self, path: str) -> str:
        """Resolve a path against the working dir (absolute paths pass through)."""
        return path if posixpath.isabs(path) else posixpath.normpath(posixpath.join(self._working_dir, path))

    async def _mkdir(self, path: str) -> None:
        await self._raw_exec(f"mkdir -p {shlex.quote(path)}", cwd=None, timeout=30)

    def _scratch_dir(self) -> str:
        """A unique, hidden staging dir under the working dir.

        ``copy_in`` only reliably writes under the working dir (a ``/tmp`` target
        is silently dropped by the guest), so scratch space lives there. The
        dotted prefix keeps it out of the agent's ``ls`` of its workspace, and it
        is always removed after use.
        """
        return posixpath.join(self._working_dir, f".stirrup_scratch_{os.urandom(6).hex()}")

    # -- file primitives ----------------------------------------------------

    async def read_file_bytes(self, path: str) -> bytes:
        """Read file content as bytes from the box (via ``copy_out``)."""
        if self._box is None:
            raise RuntimeError("ExecutionEnvironment not started.")
        resolved = self._resolve(path)
        if not await self.file_exists(resolved):
            raise FileNotFoundError(f"File not found: {path}")
        with tempfile.TemporaryDirectory() as d:
            await self._box.copy_out(resolved, d)  # -> d/<basename>
            local = Path(d) / posixpath.basename(resolved)
            return local.read_bytes()

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to a file in the box.

        ``copy_in`` only reliably targets paths under the working dir, and nests
        the payload in a subdirectory, so stage into a unique scratch dir under
        the working dir, locate the file with ``find``, and move it into place.
        """
        if self._box is None:
            raise RuntimeError("ExecutionEnvironment not started.")
        resolved = self._resolve(path)
        parent = posixpath.dirname(resolved) or "/"
        name = posixpath.basename(resolved)
        await self._mkdir(parent)
        scratch = self._scratch_dir()
        await self._mkdir(scratch)
        try:
            with tempfile.TemporaryDirectory() as d:
                host_file = Path(d) / name  # keep target basename so the moved name matches
                host_file.write_bytes(content)
                await self._box.copy_in(str(host_file), scratch)  # nests under scratch/
            res = await self._raw_exec(
                f'f=$(find {shlex.quote(scratch)} -type f | head -n1); mv -f "$f" {shlex.quote(resolved)}',
                cwd=None,
                timeout=120,
            )
            if res.exit_code != 0:
                raise RuntimeError(f"write_file_bytes mv failed for {path}: {res.stderr}")
        finally:
            await self._raw_exec(f"rm -rf {shlex.quote(scratch)}", cwd=None, timeout=30)

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in the box."""
        resolved = self._resolve(path)
        res = await self._raw_exec(f"test -f {shlex.quote(resolved)}", cwd=None, timeout=30)
        return res.exit_code == 0

    async def is_directory(self, path: str) -> bool:
        """Check if a path is a directory in the box."""
        resolved = self._resolve(path)
        res = await self._raw_exec(f"test -d {shlex.quote(resolved)}", cwd=None, timeout=30)
        return res.exit_code == 0

    async def list_files(self, path: str) -> list[str]:
        """List all files recursively under a directory in the box (relative paths)."""
        resolved = self._resolve(path)
        if not await self.is_directory(resolved):
            return []
        res = await self._raw_exec(f"find {shlex.quote(resolved)} -type f", cwd=None, timeout=120)
        if res.exit_code != 0:
            return []
        files: list[str] = []
        prefix = resolved.rstrip("/") + "/"
        for line in res.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            files.append(line[len(prefix):] if line.startswith(prefix) else line)
        return files

    # -- bulk upload (tarball, structure-preserving) ------------------------

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: "CodeExecToolProvider | None" = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        """Upload local files/dirs into the box.

        Local uploads are batched into a single tarball and untarred into the
        destination, which both preserves directory structure and sidesteps the
        ``copy_in`` ``extracted/`` nesting. Cross-environment transfers
        (``source_env`` set) defer to the base implementation.
        """
        if self._box is None:
            raise RuntimeError("ExecutionEnvironment not started.")
        if source_env is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

        dest = (dest_dir if dest_dir and posixpath.isabs(dest_dir) else self._resolve(dest_dir or "")) or self._working_dir
        await self._mkdir(dest)
        result = UploadFilesResult()

        sources: list[Path] = []
        for p in paths:
            sp = Path(p)
            if not sp.exists():
                result.failed[str(sp)] = "File or directory does not exist"
                logger.warning("Upload source does not exist: %s", sp)
                continue
            sources.append(sp)
        if not sources:
            return result

        scratch = self._scratch_dir()
        await self._mkdir(scratch)
        try:
            with tempfile.TemporaryDirectory() as d:
                tar_path = Path(d) / "payload.tar"
                with tarfile.open(tar_path, "w") as tar:
                    for sp in sources:
                        tar.add(str(sp), arcname=sp.name)  # top-level entries keep their basename
                await self._box.copy_in(str(tar_path), scratch)  # nests under scratch/
            res = await self._raw_exec(
                f't=$(find {shlex.quote(scratch)} -type f -name "*.tar" | head -n1); '
                f"tar -xf \"$t\" -C {shlex.quote(dest)}",
                cwd=None,
                timeout=600,
            )
            if res.exit_code != 0:
                for sp in sources:
                    result.failed[str(sp)] = f"tar extract failed: {res.stderr[-500:]}"
                return result
            for sp in sources:
                result.uploaded.append(
                    UploadedFile(source_path=sp, dest_path=f"{dest}/{sp.name}", size=sp.stat().st_size if sp.is_file() else 0)
                )
        finally:
            await self._raw_exec(f"rm -rf {shlex.quote(scratch)}", cwd=None, timeout=30)

        return result

    # -- images -------------------------------------------------------------

    async def view_image(self, path: str) -> ImageContentBlock:
        """Read and return an image file from the box."""
        if not path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")):
            raise ValueError(f"Unsupported image type for `{path}`.")
        file_bytes = await self.read_file_bytes(path)
        return ImageContentBlock(data=file_bytes)
