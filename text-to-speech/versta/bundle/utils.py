from shutil import rmtree, copy2
from pathlib import Path
from typing import List

def copy_folder(src: Path, dest: Path):
    """
    Copies the contents of the source directory to the destination directory.

    Args:
        src (Path): Source directory path.
        dest (Path): Destination directory
    """
    if src.exists() and src.is_dir():
        dest = dest / src.name
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dest_item = dest / item.name
            if item.is_dir():
                copy_folder(item, dest)
            else:
                copy2(item, dest_item)

def copy_folders(src_dir: Path, dest: Path):
    """
    Copies the contents of multiple source directories to the destination directory.

    Args:
        src_dir (List[Path]): List of source directory paths.
        dest (Path): Destination directory
    """
    copy_folder(src_dir, dest)

def remove_folder(dir: Path):
    """
    Removes the specified directory and all its contents.

    Args:
        dir (Path): Directory path to be removed.
    """
    if dir.exists() and dir.is_dir():
        rmtree(dir)