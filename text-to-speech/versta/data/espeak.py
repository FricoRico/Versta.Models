from shutil import move
from subprocess import run
from pathlib import Path

def build_data(source_dir: Path, build_dir: Path, output_dir: Path) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Directory {source_dir} does not exist.")

    build_output_path = build_dir / "espeak-data"

    configure_command = [
        "cmake",
        "-B" + build_output_path.as_posix(),
        source_dir
    ]

    build_command = [
        "cmake",
        "--build", build_output_path.as_posix(),
        "--target", "data"
    ]

    run(configure_command, check=True)
    print("CMake configuration completed successfully.")

    run(build_command, check=True)
    print("CMake build completed successfully.")

    move(build_output_path / "espeak-ng-data", output_dir)

    return output_dir