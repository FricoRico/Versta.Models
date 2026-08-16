from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
from typing import Tuple


def download_omnidocbench_json() -> Path:
    """Download OmniDocBench.json from HuggingFace."""
    json_path = hf_hub_download(
        repo_id="opendatalab/OmniDocBench",
        filename="OmniDocBench.json",
        repo_type="dataset",
        revision="v1_0",
    )
    return Path(json_path)


def get_omnidocbench_images() -> Tuple[Path, Path]:
    """Get OmniDocBench images and annotations from HuggingFace."""
    images_dir = (
        Path(
            snapshot_download(
                repo_id="opendatalab/OmniDocBench",
                repo_type="dataset",
                revision="v1_0",
                allow_patterns="images/**",
            )
        )
        / "images"
    )

    json_path = download_omnidocbench_json()

    return images_dir, json_path
