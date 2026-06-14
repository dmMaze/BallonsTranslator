import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from ballontranslator.utils.logger import logger as LOGGER


BACKENDS = ('auto', 'pip', 'uv', 'conda-pip')
ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
RAW_PROGRESS_RE = re.compile(r'^Progress\s+(\d+)\s+of\s+(\d+)$')


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


@dataclass
class _InstallerProgressState:
    download_message: str = ''
    total: int = 0
    last_downloaded: int = 0
    last_time: float = 0.0
    speed: float = 0.0


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
    index_args = ['-i', index_url] if index_url else []
    find_links = env.get('FIND_LINKS')
    find_links_args = ['-f', find_links] if find_links else []
    progress_args = _pip_progress_args(extra, env)
    python_executable = python_executable or sys.executable
    python_prefix = python_prefix or sys.prefix
    resolved_backend = resolve_backend(backend, env=env)

    if resolved_backend == 'uv':
        return [
            'uv', 'pip', 'install', '--python', python_executable,
            *reqs, *find_links_args, *index_args, *extra,
        ]
    if resolved_backend == 'conda-pip':
        return [
            'conda', 'run', '-p', python_prefix,
            python_executable, '-m', 'pip', 'install',
            *reqs, *progress_args, *find_links_args, *index_args, *extra,
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
        *progress_args,
        *find_links_args,
        *index_args,
        *extra,
    ]


def _pip_progress_args(extra: List[str], env: dict) -> List[str]:
    """Return pip progress args suitable for captured subprocess output.

    >>> _pip_progress_args([], {})
    ['--progress-bar', 'raw']
    >>> _pip_progress_args(['--progress-bar', 'off'], {})
    []
    """

    if env.get('PIP_PROGRESS_BAR'):
        return []
    for idx, arg in enumerate(extra):
        if arg == '--progress-bar':
            return []
        if arg.startswith('--progress-bar='):
            return []
    return ['--progress-bar', 'raw']


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
    resolved_backend = resolve_backend(backend, env=install_env)
    LOGGER.info(f'Using Python package installer backend: {resolved_backend}')
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
                encoding='utf-8',
                errors='replace',
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
    progress_state = _InstallerProgressState()
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
                _feed_progress_text(text, pending, progress_callback, progress_state, echo=True)
            elif process.poll() is not None:
                break
    finally:
        os.close(master_fd)
    _emit_progress_message(pending, progress_callback, progress_state, echo=True)
    return process.wait(), ''.join(captured)


def _stream_process_output(
    process: subprocess.Popen,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> str:
    captured = []
    pending = []
    progress_state = _InstallerProgressState()

    while True:
        chunk = process.stdout.read(1) if process.stdout is not None else ''
        if chunk == '':
            if process.poll() is not None:
                break
            continue
        captured.append(chunk)
        _feed_progress_text(chunk, pending, progress_callback, progress_state, echo=True)
    _emit_progress_message(pending, progress_callback, progress_state, echo=True)
    return ''.join(captured)


def _feed_progress_text(
    text: str,
    pending: List[str],
    progress_callback: Optional[Callable[[dict], None]] = None,
    progress_state: Optional[_InstallerProgressState] = None,
    echo: bool = False,
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
            _emit_progress_message(pending, progress_callback, progress_state, echo=echo)
        else:
            pending.append(char)
            if len(pending) >= 200:
                _emit_progress_message(pending, progress_callback, progress_state, echo=echo)


def _print_stream_text(text: str):
    try:
        print(text, end='', flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        safe_text = text.encode(encoding, errors='replace').decode(encoding, errors='replace')
        print(safe_text, end='', flush=True)


def _emit_progress_message(
    pending: List[str],
    progress_callback: Optional[Callable[[dict], None]] = None,
    progress_state: Optional[_InstallerProgressState] = None,
    echo: bool = False,
):
    if not pending:
        return
    message = ANSI_ESCAPE_RE.sub('', ''.join(pending)).strip()
    pending.clear()
    if not message:
        return
    progress_state = progress_state or _InstallerProgressState()
    progress_payload = _package_download_progress_payload(message, progress_state)
    if progress_payload is not None:
        if echo:
            if progress_payload.get('event') == 'package_download_progress':
                _print_stream_text(_format_download_progress_line(progress_payload) + '\n')
            else:
                _print_stream_text(progress_payload.get('message', message) + '\n')
        if progress_callback is not None:
            progress_callback(progress_payload)
        return

    if echo:
        _print_stream_text(message + '\n')
    if progress_callback is not None:
        progress_callback({'event': 'package_output', 'message': message})


def _package_download_progress_payload(
    message: str,
    state: _InstallerProgressState,
) -> Optional[dict]:
    """Convert pip raw progress lines into structured download progress.

    >>> state = _InstallerProgressState()
    >>> _package_download_progress_payload('Downloading https://host/torch.whl (100 MB)', state)
    {'event': 'package_output', 'message': 'Downloading torch'}
    >>> payload = _package_download_progress_payload('Progress 50 of 100', state)
    >>> (payload['event'], payload['downloaded'], payload['total'])
    ('package_download_progress', 50, 100)
    """

    if message.startswith('Downloading '):
        state.download_message = _download_display_message(message)
        return {'event': 'package_output', 'message': state.download_message}

    match = RAW_PROGRESS_RE.match(message)
    if match is None:
        return None

    downloaded = int(match.group(1))
    total = int(match.group(2))
    now = time.monotonic()
    if (
        downloaded < state.last_downloaded
        or total != state.total
        or not state.last_time
    ):
        state.speed = 0.0
        state.last_time = now
    else:
        elapsed = max(now - state.last_time, 1e-6)
        delta = max(downloaded - state.last_downloaded, 0)
        instant_speed = delta / elapsed
        if instant_speed > 0:
            state.speed = instant_speed if not state.speed else (state.speed * 0.7 + instant_speed * 0.3)
        state.last_time = now

    state.total = total
    state.last_downloaded = downloaded
    eta = None
    if total and state.speed > 0 and downloaded < total:
        eta = max(int(round((total - downloaded) / state.speed)), 0)

    return {
        'event': 'package_download_progress',
        'message': state.download_message or 'Downloading package',
        'downloaded': downloaded,
        'total': total or None,
        'speed': state.speed or None,
        'eta': eta,
    }


def _format_download_progress_line(payload: dict) -> str:
    """Return a compact terminal line for structured package download progress.

    >>> _format_download_progress_line({'message': 'Downloading a.whl', 'downloaded': 50, 'total': 100, 'speed': 10, 'eta': 5})
    'Downloading a.whl | 50.0% | 10.0 B/s | ETA 0:00:05'
    """

    downloaded = payload.get('downloaded') or 0
    total = payload.get('total')
    parts = [payload.get('message') or 'Downloading package']
    if total:
        percent = downloaded / total * 100
        parts.append(f'{percent:.1f}%')
    else:
        parts.append(_sizeof_fmt(downloaded))
    speed = payload.get('speed')
    if speed:
        parts.append(f'{_sizeof_fmt(speed)}/s')
    eta = payload.get('eta')
    if eta is not None:
        parts.append(f'ETA {_format_duration(eta)}')
    return ' | '.join(parts)


def _download_display_message(message: str) -> str:
    """Shorten pip ``Downloading`` lines for UI display.

    >>> _download_display_message('Downloading https://host/a/b/torch-1.whl (100 MB)')
    'Downloading torch'
    >>> _download_display_message('Downloading torch-1.whl.metadata (1.0 kB)')
    'Downloading torch'
    """

    target = message[len('Downloading '):].strip()
    target = re.sub(r'\s+\([^)]*\)\s*$', '', target)
    parsed = urlparse(target)
    if parsed.scheme and parsed.path:
        target = unquote(parsed.path.rsplit('/', 1)[-1])
    return f'Downloading {_simple_package_name_from_download_target(target)}'


def _simple_package_name_from_download_target(target: str) -> str:
    """Extract a readable package name from a pip download target.

    >>> _simple_package_name_from_download_target('torch-2.10.0+cu128-cp312-cp312-win_amd64.whl')
    'torch'
    >>> _simple_package_name_from_download_target('opencv_python-4.11.0.86.tar.gz')
    'opencv_python'
    """

    name = (target or '').strip()
    name = re.sub(r'\.metadata$', '', name)
    name = re.sub(r'(\.tar\.gz|\.zip|\.whl|\.tgz|\.tar\.bz2)$', '', name, flags=re.IGNORECASE)
    match = re.match(r'(.+?)-(?=\d)', name)
    if match:
        name = match.group(1)
    return name or 'package'


def _sizeof_fmt(size, suffix='B') -> str:
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'


def _format_duration(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'
