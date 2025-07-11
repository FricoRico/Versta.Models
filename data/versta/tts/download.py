from os import environ, path, listdir
from shutil import rmtree, move
from subprocess import run
from pathlib import Path
from tarfile import open as taropen
from requests import get

from .utils import copy_folder

def download_folder_from_git(export_path: Path, download_path: Path, repo_url: str, folder_path: str = None) -> Path:
    """
    Clones a Git repository using SSH and extracts a specific folder to an output directory.

    Args:
        export_path (Path): The directory where the extracted folder will be saved.
        download_path (Path): The directory where the repository will be cloned.
        repo_url (str): The SSH URL of the Git repository.
        folder_path (str): The name of the folder to extract from the repository.

    Returns:
        Path: The path to the extracted folder.
    """
    print("Downloading folder from Git repository...")

    repo_name = repo_url.split('/')[-1].replace('.git', '')

    clone_path = download_path / repo_name
    if clone_path.exists():
        rmtree(clone_path)

    clone_path.mkdir(parents=True)

    env = environ.copy()
    run(['git', 'clone', repo_url, clone_path], check=True, env=env)

    if folder_path is None:
        source_path = clone_path
    else:
        source_path = clone_path / folder_path
    if not path.exists(source_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist in the repository.")

    if folder_path is None:
        target_path = export_path
    else:
        target_path = export_path / folder_path
    if target_path.exists():
        rmtree(target_path)

    copy_folder(source_path, export_path)

    if folder_path is None:
        return export_path / repo_name

    return target_path

def download_folder_from_tarball(export_path: Path, download_path: Path, tarball_url: str, folder_path: str) -> Path:
    """
    Downloads a tarball from a URL and extracts a specific folder to an output directory.

    Args:
        export_path (Path): The directory where the extracted folder will be saved.
        download_path (Path): The directory where the tarball will be temporarily saved.
        tarball_url (str): The URL of the tarball.
        folder_path (str): The name of the folder to extract from the tarball.

    Returns:
        Path: The path to the extracted folder.
    """
    print("Downloading tarball from URL...")

    tarball_name = tarball_url.split('/')[-1]
    tarball_path = download_path / tarball_name

    print(f"Downloading {tarball_url} to {tarball_path}...")

    response = get(tarball_url, stream=True)
    response.raise_for_status()  # Ensure we got a successful response

    print("Saving tarball to disk...")
    with open(tarball_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting tarball...")
    with taropen(tarball_path, 'r:*') as tar:
        tar.extractall(path=download_path)

    source_path = Path(tarball_path.as_posix().replace(".tar.gz", ""))
    if not source_path.exists():
        raise FileNotFoundError(f"The folder '{tarball_name.split('.')[0]}' does not exist in the tarball.")

    target_path = export_path / folder_path
    if target_path.exists():
        rmtree(target_path)

    target_path.mkdir(parents=True)

    for item in listdir(source_path):
        source_file = path.join(source_path, item)
        target_file  = path.join(target_path, item)
        move(source_file, target_file)

    return target_path