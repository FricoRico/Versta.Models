from pathlib import Path

with open(Path(__file__).parent / ".." / "version.txt", "r") as _version_file:
    VERSION = _version_file.read().strip()
