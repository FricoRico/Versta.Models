from pathlib import Path
from shutil import copy

def convert_sentence_piece_files(model_dir: Path):
    """
    Converts the sentence piece file names if they are not already called 'source.spm' and 'target.spm'. If more than
    one sentence piece file is found, that is not named 'source.spm' or 'target.spm', an error is raised.

    If a single sentence piece file is found, it is renamed to 'source.spm' and 'target.spm' is created as a copy of it.

    Args:
        model_dir (Path): Path to the directory containing the pre-trained model.
    """
    spm_files = list(model_dir.glob("*.spm"))

    if len(spm_files) == 0:
        raise FileNotFoundError("No sentence piece files found in the model directory.")

    if any(file.name in ["source.spm", "target.spm"] for file in spm_files):
        return

    if len(spm_files) > 1:
        raise ValueError("Multiple sentence piece files found, but none are named 'source.spm' or 'target.spm'. "
                         "Please ensure the model directory contains only one sentence piece file or rename it.")

    print(f"Renaming sentence piece file {spm_files[0].name} to 'source.spm' and creating 'target.spm' as a copy...")

    copy(spm_files[0], model_dir / "target.spm")
    copy(spm_files[0], model_dir / "source.spm")