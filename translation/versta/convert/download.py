from zipfile import ZipFile
from pathlib import Path
import requests

def download_model(model_uri: str, download_dir: Path) -> Path:
    """
    Download the model files from the provided URL.

    Args:
        model (str): URL to the model file.
        download_dir (Path): Directory to save the downloaded model files.
    """
    file_dir = _download_zip(model_uri, download_dir)
    extract_dir = _extract_zip(file_dir, download_dir)

    return extract_dir

def _download_zip(model_uri: str, output_dir: Path) -> Path:
    """
    Download a zip file from the specified URL to the output directory.

    Args:
        model_uri (str): URL to the zip file to download.
        output_dir (Path): Path to the directory where the zip file will be saved
    """
    file_name = model_uri.split("/")[-1]
    file_path = output_dir / file_name

    print(f"Downloading {model_uri} to {file_path}")

    with requests.get(model_uri, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return file_path

def _extract_zip(zip_file: Path, output_dir: Path) -> Path:
    """
    Extract the contents of a zip file to the specified output directory.

    Args:
        zip_file (Path): Path to the zip file to extract.
        output_dir (Path): Path to the directory where the zip file will be extracted
    """
    extract_path = output_dir / zip_file.stem
    extract_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_file} to {extract_path}")

    with ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path