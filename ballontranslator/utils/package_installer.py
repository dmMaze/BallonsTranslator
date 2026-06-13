import os
import re
import select
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from ballontranslator.utils.logger import logger as LOGGER


BACKENDS = ('auto', 'pip', 'uv', 'conda-pip')
ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


@dataclass
class InstallResult:
    """Structured result from a package install command.

    >>> result = InstallResult(True, ['python', '-m', 'pip', 'install', 'torch'])
    >>> result.ok
    True
    >>> result.command_text
    'python -m pip install torch'
    """

    ok: bool
    command: List[str]
    returncode: int = 0
    stdout: str = ''
    stderr: str = ''
    error: str = ''

    @property
    def command_text(self) -> str:
        return shlex.join(self.command)


def resolve_backend(backend: str = 'auto', env: Optional[dict] = None) -> str:
    """Resolve the installer backend used for command generation.

    >>> resolve_backend('pip')
    'pip'
    >>> resolve_backend('unknown')
    'auto'
    """

    if backend != 'auto':
        return backend if backend in BACKENDS else 'auto'
    env = env or os.environ
    if shutil.which('uv', path=env.get('PATH')):
        return 'uv'
    return 'pip'


def build_install_command(
    requirements: Iterable[str] = (),
    requirements_file: str = '',
    backend: str = 'auto',
    extra_args: str = '',
    env: Optional[dict] = None,
    python_executable: str = '',
    python_prefix: str = '',
) -> List[str]:
    """Build an install command without using ``shell=True``.

    >>> build_install_command(['openai>=2.8.1'], backend='pip', python_executable='python')[:5]
    ['python', '-m', 'pip', 'install', 'openai>=2.8.1']
    >>> build_install_command(['betterproto'], backend='uv', python_executable='python')[:5]
    ['uv', 'pip', 'install', '--python', 'python']
    >>> build_install_command(['torch', 'torch'], backend='pip', python_executable='python').count('torch')
    1
    """

    reqs = [req for req in dict.fromkeys(requirements) if req]
    if requirements_file:
        reqs.extend(['-r', requirements_file])
    extra = shlex.split(extra_args or '')
    env = env or os.environ
    index_url = env.get('INDEX_URL')
    index_args = ['--index-url', index_url] if index_url else []
    python_executable = python_executable or sys.executable
    python_prefix = python_prefix or sys.prefix
    resolved_backend = resolve_backend(backend, env=env)

    if resolved_backend == 'uv':
        return [
            'uv', 'pip', 'install', '--python', python_executable,
            *reqs, *index_args, *extra,
        ]
    if resolved_backend == 'conda-pip':
        return [
            'conda', 'run', '-p', python_prefix,
            python_executable, '-m', 'pip', 'install',
            *reqs, *index_args, *extra,
        ]
    return [
        python_executable,
        '-m',
        'pip',
        'install',
        *reqs,
        '--prefer-binary',
        '--disable-pip-version-check',
        '--no-warn-script-location',
        *index_args,
        *extra,
    ]


def install(
    requirements: Iterable[str] = (),
    requirements_file: str = '',
    backend: str = 'auto',
    extra_args: str = '',
    env: Optional[dict] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> InstallResult:
    """Install Python packages and stream installer output.

    The function is intentionally side-effectful, so doctests cover the empty
    command construction path through a mocked backend rather than running pip.

    >>> cmd = build_install_command([], backend='pip', python_executable='python')
    >>> cmd[:4]
    ['python', '-m', 'pip', 'install']
    """

    install_env = env or os.environ.copy()
    command = build_install_command(
        requirements=requirements,
        requirements_file=requirements_file,
        backend=backend,
        extra_args=extra_args,
        env=install_env,
    )
    index_url = install_env.get('INDEX_URL')
    if index_url:
        LOGGER.info(f'Using PyPI package mirror for package install: {index_url}')
    if _can_stream_with_pty():
        try:
            returncode, output = _run_with_pty(command, env=install_env, progress_callback=progress_callback)
        except Exception as e:
            return InstallResult(False, command, error=str(e), returncode=-1)
    else:
        try:
            process = subprocess.Popen(
                command,
                env=install_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
                bufsize=1,
            )
        except Exception as e:
            return InstallResult(False, command, error=str(e), returncode=-1)
        output = _stream_process_output(process, progress_callback=progress_callback)
        returncode = process.wait()
    return InstallResult(
        returncode == 0,
        command,
        returncode=returncode,
        stdout=output,
        stderr='',
    )


def _can_stream_with_pty() -> bool:
    if os.name == 'nt':
        return False
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    try:
        import pty  # noqa: F401
    except Exception:
        return False
    return True


def _run_with_pty(
    command: List[str],
    env: Optional[dict] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> Tuple[int, str]:
    import pty

    master_fd, slave_fd = pty.openpty()
    try:
        process = subprocess.Popen(
            command,
            env=env or os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            shell=False,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    captured = []
    pending = []
    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                text = chunk.decode(errors='replace')
                captured.append(text)
                print(text, end='', flush=True)
                _feed_progress_text(text, pending, progress_callback)
            elif process.poll() is not None:
                break
    finally:
        os.close(master_fd)
    _emit_progress_message(pending, progress_callback)
    return process.wait(), ''.join(captured)


def _stream_process_output(
    process: subprocess.Popen,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> str:
    captured = []
    pending = []

    while True:
        chunk = process.stdout.read(1) if process.stdout is not None else ''
        if chunk == '':
            if process.poll() is not None:
                break
            continue
        captured.append(chunk)
        print(chunk, end='', flush=True)
        _feed_progress_text(chunk, pending, progress_callback)
    _emit_progress_message(pending, progress_callback)
    return ''.join(captured)


def _feed_progress_text(
    text: str,
    pending: List[str],
    progress_callback: Optional[Callable[[dict], None]] = None,
):
    """Split installer output into progress callback messages.

    >>> events = []
    >>> pending = []
    >>> _feed_progress_text('Downloading 1 MB/s\\nDone\\n', pending, events.append)
    >>> [event['message'] for event in events]
    ['Downloading 1 MB/s', 'Done']
    """

    for char in text:
        if char in {'\n', '\r'}:
            _emit_progress_message(pending, progress_callback)
        else:
            pending.append(char)
            if len(pending) >= 200:
                _emit_progress_message(pending, progress_callback)


def _emit_progress_message(
    pending: List[str],
    progress_callback: Optional[Callable[[dict], None]] = None,
):
    if not pending:
        return
    message = ANSI_ESCAPE_RE.sub('', ''.join(pending)).strip()
    pending.clear()
    if message and progress_callback is not None:
        progress_callback({'event': 'package_output', 'message': message})
