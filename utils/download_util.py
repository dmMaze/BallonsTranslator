import math
import os
import requests
import traceback
import re
import sys
import shutil
import os.path as osp
from typing import List, Union
import hashlib

from tqdm import tqdm
from urllib.parse import urlparse
from torch.hub import download_url_to_file as _torchhub_download_url_to_file, get_dir
import requests
import tqdm
from py7zr import pack_7zarchive, unpack_7zarchive
import ssl

from . import shared
from .logger import logger as LOGGER

shutil.register_archive_format('7zip', pack_7zarchive, description='7zip archive')
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest().lower()


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file size.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'


def download_file_from_google_drive(file_id, save_path):
    """Download files from google drive.

    Ref:
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive  # noqa E501

    Args:
        file_id (str): File id.
        save_path (str): Save path.
    """

    session = requests.Session()
    URL = 'https://docs.google.com/uc?export=download'
    params = {'id': file_id, 'confirm': 't'}    # https://stackoverflow.com/a/73893665/17671327

    response = session.get(URL, params=params, stream=True)
    token = get_confirm_token(response)
    if token:
        params['confirm'] = token
        response = session.get(URL, params=params, stream=True)

    # get file size
    response_file_size = session.get(URL, params=params, stream=True, headers={'Range': 'bytes=0-2'})
    if 'Content-Range' in response_file_size.headers:
        file_size = int(response_file_size.headers['Content-Range'].split('/')[1])
    else:
        file_size = None

    save_response_content(response, save_path, file_size)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, file_size=None, chunk_size=32768):
    if file_size is not None:
        pbar = tqdm(total=math.ceil(file_size / chunk_size), unit='chunk')

        readable_file_size = sizeof_fmt(file_size)
    else:
        pbar = None

    with open(destination, 'wb') as f:
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size):
            downloaded_size += chunk_size
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f'Download {sizeof_fmt(downloaded_size)} / {readable_file_size}')
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
        if pbar is not None:
            pbar.close()

# def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
#     """Load file form http url, will download models if necessary.

#     Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

#     Args:
#         url (str): URL to be downloaded.
#         model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
#             Default: None.
#         progress (bool): Whether to show the download progress. Default: True.
#         file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

#     Returns:
#         str: The path to the downloaded file.
#     """
#     if model_dir is None:  # use the pytorch hub_dir
#         hub_dir = get_dir()
#         model_dir = os.path.join(hub_dir, 'checkpoints')

#     os.makedirs(model_dir, exist_ok=True)

#     parts = urlparse(url)
#     filename = os.path.basename(parts.path)
#     if file_name is not None:
#         filename = file_name
#     cached_file = os.path.abspath(os.path.join(model_dir, filename))
#     if not os.path.exists(cached_file):
#         print(f'Downloading: "{url}" to {cached_file}\n')
#         download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
#     return cached_file


def torchhub_download_url_to_file(url: str, dst: str, progress: bool = True):
    original_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context  # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org
    _torchhub_download_url_to_file(url, dst, progress=progress)
    ssl._create_default_https_context = original_ctx

def check_local_file(local_file: str, sha256_precal: str = None, cache_hash: bool = False):

    file_exists = osp.exists(local_file)
    valid_hash, sha256_calculated = True, sha256_precal

    if file_exists and sha256_precal is not None and shared.check_local_file_hash:
        sha256_precal = sha256_precal.lower()
        if cache_hash and local_file in shared.cache_data and shared.cache_data[local_file].lower() == sha256_precal:
            pass
        else:
            sha256_calculated = calculate_sha256(local_file).lower()
            if sha256_calculated != sha256_precal:
                valid_hash = False
            if cache_hash:
                shared.cache_data[local_file] = sha256_calculated
                shared.CACHE_UPDATED = True
    
    return file_exists, valid_hash, sha256_calculated


def get_filename_from_url(url: str, default: str = '') -> str:
    m = re.search(r'/([^/?]+)[^/]*$', url)
    if m:
        return m.group(1)
    return default


def download_url_with_progressbar(url: str, path: str,):
    if os.path.basename(path) in ('.', '') or os.path.isdir(path):
        new_filename = get_filename_from_url(url)
        if not new_filename:
            raise Exception('Could not determine filename')
        path = os.path.join(path, new_filename)

    headers = {}
    downloaded_size = 0
    # the resume downloading here is buggy when the local file is corrupted or over-sized or intended to be replaced
    # if os.path.isfile(path):  # its actually buggy
    #     downloaded_size = os.path.getsize(path)
    #     headers['Range'] = 'bytes=%d-' % downloaded_size
    #     headers['Accept-Encoding'] = 'deflate'

    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
    if downloaded_size and r.headers.get('Accept-Ranges') != 'bytes':
        print('Error: Webserver does not support partial downloads. Restarting from the beginning.')
        r = requests.get(url, stream=True, allow_redirects=True)
        downloaded_size = 0
    total = int(r.headers.get('content-length', 0))
    chunk_size = 1024

    if r.ok:
        with tqdm.tqdm(
            desc=os.path.basename(path),
            initial=downloaded_size,
            total=total+downloaded_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            with open(path, 'ab' if downloaded_size else 'wb') as f:
                is_tty = sys.stdout.isatty()
                downloaded_chunks = 0
                for data in r.iter_content(chunk_size=chunk_size):
                    size = f.write(data)
                    bar.update(size)

                    # Fallback for non TTYs so output still shown
                    downloaded_chunks += 1
                    if not is_tty and downloaded_chunks % 1000 == 0:
                        print(bar)
    else:
        raise Exception(f'Couldn\'t resolve url: "{url}" (Error: {r.status_code})')



def try_download_files(url: str, 
                        files: List[str], 
                        save_files = List[str], 
                        sha256_pre_calculated: List[str] = None, 
                        concatenate_url_filename: int = 0,
                        cache_hash: bool = False,
                        download_method: str = '',
                        gdrive_file_id: str = None):
    
    all_successful = True
    
    for file, savep, sha256_precal in zip(files, save_files, sha256_pre_calculated):
        save_dir = osp.dirname(savep)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        file_exists, valid_hash, sha256_calculated = check_local_file(savep, sha256_precal, cache_hash=cache_hash)
        if file_exists:
            if valid_hash:
                continue
            else:
                LOGGER.warning(f'Mismatch between local file {savep} and pre-calculated hash: "{sha256_calculated}" <-> "{sha256_precal.lower()}", it will be redownloaded...')
        
        try:
            if concatenate_url_filename == 1:
                download_url = url + file
            elif concatenate_url_filename == 2:
                download_url = url + osp.basename(file)
            else:
                download_url = url

            if gdrive_file_id is not None:
                download_file_from_google_drive(gdrive_file_id, savep)
            elif download_method == 'torch_hub':
                LOGGER.info(f'downloading {savep} from {download_url} ...')
                torchhub_download_url_to_file(download_url, savep)
            else:
                download_url_with_progressbar(download_url, savep)
            file_exists, valid_hash, sha256_calculated = check_local_file(savep, sha256_precal, cache_hash=cache_hash)
            if not file_exists:
                raise Exception(f'Some how the downloaded {savep} doesnt exists.')
            elif not valid_hash:
                raise Exception(f'Mismatch between newly downloaded {savep} and pre-calculated hash: "{sha256_calculated}" <-> "{sha256_precal.lower()}"')

        except:
            err_msg = traceback.format_exc()
            all_successful = False
            LOGGER.error(err_msg)
            LOGGER.error(f'Failed downloading {file} from {download_url}, please manually save it to {savep}')
    
    return all_successful


def download_and_check_files(url: str, 
                        files: Union[str, List], 
                        save_files = None, 
                        sha256_pre_calculated: Union[str, List] = None, 
                        concatenate_url_filename: int = 0, 
                        archived_files: List = None, 
                        archive_sha256_pre_calculated: Union[str, List] = None,
                        save_dir: str = None,
                        download_method: str = 'torch_hub',
                        gdrive_file_id: str = None):
        
    def _wrap_up_checkinputs(files: Union[str, List], save_files: Union[str, List] = None, sha256_pre_calculated: Union[str, List] = None, save_dir: str = None):
        '''
        ensure they're lists with equal length
        '''
        if not isinstance(files, List):
            files = [files]
        if not isinstance(sha256_pre_calculated, List):
            if sha256_pre_calculated is None:
                sha256_pre_calculated = [None] * len(files)
            else:
                sha256_pre_calculated = [sha256_pre_calculated]
        if save_files is None:
            save_files = files
        elif not isinstance(save_files, List):
            save_files = [save_files]

        assert len(files) == len(sha256_pre_calculated) == len(save_files)

        if save_dir is not None:
            _save_files = []
            for savep in save_files:
                _save_files.append(osp.join(save_dir, savep))
            save_files = _save_files

        return files, save_files, sha256_pre_calculated
    
    def _all_valid(save_files: List[str] = None, sha256_pre_calculated: List[str] = None,):
        for savep, sha256_precal in zip(save_files, sha256_pre_calculated):
            file_exists, valid_hash, sha256_calculated = check_local_file(savep, sha256_precal, cache_hash=True)
            if not file_exists or not valid_hash:
                return False
        return True
    
        
    files, save_files, sha256_pre_calculated = _wrap_up_checkinputs(files, save_files, sha256_pre_calculated, save_dir)

    if archived_files is None:
        return try_download_files(url, files, save_files, sha256_pre_calculated, concatenate_url_filename, cache_hash=True, download_method=download_method, gdrive_file_id=gdrive_file_id)

    # handle archived
    if _all_valid(save_files, sha256_pre_calculated):
        return [], None
    
    if isinstance(archived_files, str):
        archived_files = [archived_files]

    # download archive files
    tmp_downloaded_archives = [osp.join(shared.cache_dir, archive_name) for archive_name in archived_files]
    _, _, archive_sha256_pre_calculated = _wrap_up_checkinputs(archived_files, tmp_downloaded_archives, archive_sha256_pre_calculated)
    archive_downloaded = try_download_files(url, archived_files, tmp_downloaded_archives, archive_sha256_pre_calculated, concatenate_url_filename, cache_hash=False, download_method=download_method, gdrive_file_id=gdrive_file_id)
    if not archive_downloaded:
        return False
    
    # extract archived
    archivep = tmp_downloaded_archives[0] # todo: support multi-volume
    extract_dir = osp.join(shared.cache_dir, 'tmp_extract')
    os.makedirs(extract_dir, exist_ok=True)
    LOGGER.info(f'Extracting {archivep} ...')
    shutil.unpack_archive(archivep, extract_dir)

    all_valid = True
    for file, savep, sha256_precal in zip(files, save_files, sha256_pre_calculated):
        unarchived = osp.join(extract_dir, file)
        save_dir = osp.dirname(savep)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        shutil.move(unarchived, savep)
        file_exists, valid_hash, sha256_calculated = check_local_file(savep, sha256_precal, cache_hash=True)
        if not file_exists:
            LOGGER.error(f'The unarchived file {savep} doesnt exists.')
            all_valid = False
        elif not valid_hash:
            LOGGER.error(f'Mismatch between the unarchived {savep} and pre-calculated hash: "{sha256_calculated}" <-> "{sha256_precal.lower()}"')
            all_valid = False

    if all_valid:
        # clean archive files
        shutil.rmtree(extract_dir)
        for p in tmp_downloaded_archives:
            os.remove(p)

    return all_valid
