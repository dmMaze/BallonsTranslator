import json
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from . import shared
from .logger import logger as LOGGER
from .version import get_current_version


RELEASES_URL = 'https://github.com/dmMaze/BallonsTranslator/releases'
LATEST_RELEASE_API_URL = 'https://api.github.com/repos/dmMaze/BallonsTranslator/releases/latest'
UPDATE_BRANCH = 'userspace_update'
SOURCE_UPDATE_DIRS = ('ballontranslator', 'resources')
SOURCE_UPDATE_FILES = ('pyproject.toml', 'requirements.txt')


@dataclass
class ReleaseInfo:
    tag_name: str
    version: str
    html_url: str
    zip_url: str
    name: str = ''
    body: str = ''
    published_at: str = ''


@dataclass
class UpdateResult:
    status: str
    current_version: str
    latest_version: str
    release_url: str = ''
    zip_path: str = ''
    git_message: str = ''
    release_info: Optional[ReleaseInfo] = None


def normalize_version_tag(version: str) -> str:
    """Normalize a release tag for version comparison and filenames.

    >>> normalize_version_tag('v1.4.1')
    '1.4.1'
    >>> normalize_version_tag(' release-2.0 ')
    '2.0'
    """

    value = version.strip()
    for prefix in ('release-', 'Release-', 'v', 'V'):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def version_sort_key(version: str):
    """Return a conservative comparable key for dotted app versions.

    >>> version_sort_key('1.4.10') > version_sort_key('1.4.2')
    True
    """

    parts = []
    for token in normalize_version_tag(version).replace('-', '.').split('.'):
        if token.isdigit():
            parts.append((0, int(token)))
        else:
            parts.append((1, token))
    return tuple(parts)


def is_remote_newer(local_version: str, remote_version: str) -> bool:
    """Return whether ``remote_version`` should update ``local_version``.

    >>> is_remote_newer('1.4.0', 'v1.4.1')
    True
    >>> is_remote_newer('1.4.1', 'v1.4.1')
    False
    """

    try:
        from packaging.version import Version

        return Version(normalize_version_tag(remote_version)) > Version(normalize_version_tag(local_version))
    except Exception:
        return version_sort_key(remote_version) > version_sort_key(local_version)


def format_git_update_message(update_branch: str, local_changes_saved: bool, local_branch: str = '') -> str:
    """Return the user-facing git safety note for the update dialog.

    >>> print(format_git_update_message('userspace_update', True, 'dev'))
    Local git changes on branch "dev" were saved before updating.
    The update was applied on branch: userspace_update
    To restore your local changes, run:
    git switch dev
    git stash pop
    """

    lines = [
        f'The update was applied on branch: {update_branch}',
    ]
    if local_changes_saved:
        if local_branch:
            lines.insert(0, f'Local git changes on branch "{local_branch}" were saved before updating.')
        else:
            lines.insert(0, 'Local git changes were saved before updating.')
        lines.extend([
            'To restore your local changes, run:',
            f'git switch {local_branch or update_branch}',
            'git stash pop',
        ])
    else:
        lines.insert(0, 'A git repository was detected.')
    return '\n'.join(lines)


def release_info_from_api_payload(payload: dict) -> ReleaseInfo:
    """Build release metadata from the GitHub release API response.

    >>> info = release_info_from_api_payload({'tag_name': 'v1.4.1', 'html_url': 'u', 'zipball_url': 'z', 'body': 'notes'})
    >>> (info.version, info.zip_url, info.body)
    ('1.4.1', 'z', 'notes')
    """

    tag_name = payload.get('tag_name') or ''
    zip_url = payload.get('zipball_url') or ''
    html_url = payload.get('html_url') or RELEASES_URL
    if not tag_name or not zip_url:
        raise RuntimeError('GitHub release response did not include a tag or source zip URL.')
    return ReleaseInfo(
        tag_name=tag_name,
        version=normalize_version_tag(tag_name),
        html_url=html_url,
        zip_url=zip_url,
        name=payload.get('name') or tag_name,
        body=payload.get('body') or '',
        published_at=payload.get('published_at') or '',
    )


class BallonsTranslatorUpdater:
    """Check GitHub releases and replace the local application source.

    >>> updater = BallonsTranslatorUpdater(program_path='/tmp/app', cache_dir='/tmp/cache')
    >>> updater.program_path.name
    'app'
    """

    def __init__(
        self,
        program_path: str = None,
        cache_dir: str = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.program_path = Path(program_path or shared.PROGRAM_PATH).resolve()
        self.cache_dir = Path(cache_dir or shared.cache_dir).resolve()
        self.progress_callback = progress_callback

    def check_and_update(self) -> UpdateResult:
        result = self.check_latest_release()
        if result.status != 'available' or result.release_info is None:
            return result

        return self.apply_update(result.release_info, result.current_version)

    def check_latest_release(self) -> UpdateResult:
        """Check for a newer release without modifying local files.

        >>> updater = BallonsTranslatorUpdater(program_path='/tmp/app', cache_dir='/tmp/cache')
        >>> isinstance(updater.program_path, Path)
        True
        """

        current_version = get_current_version(str(self.program_path))
        release_info = self.query_latest_release()
        self._notify('compare_versions', 15, f'{current_version} -> {release_info.version}')
        if not is_remote_newer(current_version, release_info.version):
            return UpdateResult(
                status='up_to_date',
                current_version=current_version,
                latest_version=release_info.version,
                release_url=release_info.html_url,
                release_info=release_info,
            )

        LOGGER.info(f'BallonsTranslator update available: {current_version} -> {release_info.version}')
        return UpdateResult(
            status='available',
            current_version=current_version,
            latest_version=release_info.version,
            release_url=release_info.html_url,
            release_info=release_info,
        )

    def apply_update(self, release_info: ReleaseInfo, current_version: str = None) -> UpdateResult:
        current_version = current_version or get_current_version(str(self.program_path))
        zip_path = self.download_source_zip(release_info)
        git_message = self.prepare_git_worktree(release_info.version)
        self.install_source_zip(zip_path)
        self._notify('done', 100, release_info.version)
        return UpdateResult(
            status='updated',
            current_version=current_version,
            latest_version=release_info.version,
            release_url=release_info.html_url,
            zip_path=str(zip_path),
            git_message=git_message,
            release_info=release_info,
        )

    def query_latest_release(self) -> ReleaseInfo:
        self._notify('query_release', 5, RELEASES_URL)
        request = Request(
            LATEST_RELEASE_API_URL,
            headers={
                'Accept': 'application/vnd.github+json',
                'User-Agent': 'BallonsTranslator-Updater',
            },
        )
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode('utf8'))
        except (OSError, URLError, json.JSONDecodeError) as e:
            raise RuntimeError(f'Failed to query GitHub releases at {RELEASES_URL}: {e}') from e
        return release_info_from_api_payload(payload)

    def download_source_zip(self, release_info: ReleaseInfo) -> Path:
        from .download_util import download_url_to_file

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.cache_dir / f'BallonsTranslator_{release_info.version}_source.zip'
        self._notify('download_start', 20, zip_path.name)
        download_url_to_file(
            release_info.zip_url,
            str(zip_path),
            progress_callback=self._on_download_progress,
        )
        return zip_path

    def prepare_git_worktree(self, latest_version: str) -> str:
        git_dir = self.program_path / '.git'
        if not git_dir.exists() or shutil.which('git') is None:
            return ''

        local_branch = self._git(['branch', '--show-current'], check=False).stdout.strip()
        status = self._git(['status', '--porcelain']).stdout.strip()
        local_changes_saved = False
        if status:
            self._notify('git_safety', 70, 'Saving local git changes')
            self._git(['add', '-A'])
            stash_message = f'BallonsTranslator updater local changes before {latest_version}'
            stash = self._git(['stash', 'push', '-u', '-m', stash_message]).stdout.strip()
            local_changes_saved = True
            LOGGER.info(stash or f'Local changes saved to git stash: {stash_message}')

        branch_exists = self._git(['rev-parse', '--verify', UPDATE_BRANCH], check=False).returncode == 0
        if branch_exists:
            self._git(['switch', UPDATE_BRANCH])
        else:
            self._git(['switch', '-c', UPDATE_BRANCH])
        message = format_git_update_message(UPDATE_BRANCH, local_changes_saved, local_branch)
        LOGGER.info(message)
        return message

    def install_source_zip(self, zip_path: Path) -> None:
        self._notify('extract_source', 80, zip_path.name)
        extract_root = self.cache_dir / f'{zip_path.stem}_extracted'
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            self._safe_extract(archive, extract_root)

        release_root = self._find_release_root(extract_root)
        self._notify('replace_source', 90, ', '.join(SOURCE_UPDATE_DIRS + SOURCE_UPDATE_FILES))
        for dirname in SOURCE_UPDATE_DIRS:
            self._replace_directory(release_root / dirname, self.program_path / dirname)
        for filename in SOURCE_UPDATE_FILES:
            self._replace_file(release_root / filename, self.program_path / filename)
        try:
            self._cleanup_downloaded_source(zip_path, extract_root)
        except Exception as e:
            LOGGER.warning(f'Failed to clean up downloaded update source: {e}')

    def _notify(self, event: str, progress: int, message: str = '') -> None:
        LOGGER.info(f'Updater: {event} {message}')
        if self.progress_callback is not None:
            self.progress_callback({
                'event': event,
                'progress': progress,
                'message': message,
            })

    def _on_download_progress(self, payload: dict) -> None:
        event = payload.get('event')
        if event == 'file_progress':
            total = payload.get('total')
            downloaded = payload.get('downloaded') or 0
            progress = 35
            if total:
                progress = max(20, min(60, int(downloaded / total * 40) + 20))
            self._notify('download_progress', progress, os.path.basename(payload.get('path', '')))
        elif event == 'file_done':
            self._notify('download_done', 60, os.path.basename(payload.get('path', '')))

    def _git(self, args, check: bool = True):
        result = subprocess.run(
            ['git', *args],
            cwd=str(self.program_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f'git {" ".join(args)} failed: {result.stderr.strip()}')
        return result

    def _safe_extract(self, archive: zipfile.ZipFile, destination: Path) -> None:
        destination = destination.resolve()
        for member in archive.infolist():
            target_path = (destination / member.filename).resolve()
            if destination not in target_path.parents and target_path != destination:
                raise RuntimeError(f'Unsafe path in update archive: {member.filename}')
        archive.extractall(destination)

    def _find_release_root(self, extract_root: Path) -> Path:
        for candidate in extract_root.iterdir():
            if candidate.is_dir() and (candidate / 'ballontranslator').is_dir():
                return candidate
        raise RuntimeError('Downloaded source archive does not contain a ballontranslator source directory.')

    def _cleanup_downloaded_source(self, zip_path: Path, extract_root: Path) -> None:
        for path in (zip_path, extract_root):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    def _replace_directory(self, source: Path, target: Path) -> None:
        if not source.exists():
            LOGGER.info(f'Update archive no longer includes source directory; removing target: {target}')
            if target.is_dir():
                shutil.rmtree(target)
            return
        if not source.is_dir():
            raise RuntimeError(f'Update archive is missing source directory: {source}')
        # Nested update dirs must create their parent without replacing the parent itself.
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(f'.{target.name}_update_tmp')
        old_target = target.with_name(f'.{target.name}_old_tmp')
        for path in (temp_target, old_target):
            if path.exists():
                shutil.rmtree(path)
        shutil.copytree(source, temp_target)
        try:
            if target.exists():
                target.rename(old_target)
            temp_target.rename(target)
            if old_target.exists():
                shutil.rmtree(old_target)
        except Exception:
            if target.exists():
                shutil.rmtree(target)
            if old_target.exists():
                old_target.rename(target)
            raise
        finally:
            if temp_target.exists():
                shutil.rmtree(temp_target)

    def _replace_file(self, source: Path, target: Path) -> None:
        if not source.is_file():
            raise RuntimeError(f'Update archive is missing source file: {source}')
        temp_target = target.with_name(f'.{target.name}.update_tmp')
        shutil.copy2(source, temp_target)
        os.replace(temp_target, target)
