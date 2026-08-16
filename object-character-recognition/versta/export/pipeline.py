from pathlib import Path
from typing import List

from .convert_mnn import convert_to_mnn
from .convert_paddle import convert_to_onnx
from .definitions import (
    FOLD_VARIANTS,
    QUARTER_VARIANT_NOTE,
    keys_filename,
    mnn_filename,
)
from .download import download_file, extract_tar
from .fold_deconv import fold_variant_graph
from .keys import write_keys
from .manifest import file_entry
from .typing import ManifestFile, ModelSpec

DET_PRIORITIES = {"": 2, "half": 1, "quarter": 3}


def convert_model(
    spec: ModelSpec,
    work_dir: Path,
    pack_dir: Path,
    mnnconvert: Path,
) -> List[ManifestFile]:
    """
    Runs the full conversion chain for a single model: download, extract,
    paddle2onnx, MNNConvert — plus detector fold variants and recognizer keys.

    Args:
        spec (ModelSpec): The model catalog entry.
        work_dir (Path): Scratch directory for tars, extracted models and ONNX.
        pack_dir (Path): The pack output directory.
        mnnconvert (Path): The MNNConvert binary.

    Returns:
        List[ManifestFile]: Manifest entries for the produced files, in pack
        order.
    """
    downloads = work_dir / "downloads"

    if spec["kind"] == "aligner":
        # Non-Paddle models ship as ready ONNX: no tar extraction or
        # paddle2onnx step.
        onnx_path = download_file(spec["url"], downloads / f"{spec['stem']}.onnx")
        mnn_path = convert_to_mnn(mnnconvert, onnx_path, pack_dir / mnn_filename(spec))
        print(mnn_path)
        return [file_entry(mnn_path, spec["kind"], 1)]

    extracted = work_dir / "extracted"

    tar_path = download_file(spec["url"], downloads / f"{spec['stem']}.tar")
    extract_tar(tar_path, extracted)
    model_dir = extracted / spec["stem"]
    onnx_path = work_dir / "onnx" / f"{spec['stem']}.onnx"
    convert_to_onnx(spec, model_dir, onnx_path)

    entries: List[ManifestFile] = []
    mnn_path = convert_to_mnn(mnnconvert, onnx_path, pack_dir / mnn_filename(spec))
    print(mnn_path)

    if spec["kind"] == "detector":
        entries.append(file_entry(mnn_path, "detector", DET_PRIORITIES[""]))
        for variant, deconvs in FOLD_VARIANTS.items():
            folded_onnx = work_dir / "onnx" / f"{spec['stem']}_{variant}.onnx"
            fold_variant_graph(onnx_path, deconvs, folded_onnx)
            folded_mnn = convert_to_mnn(
                mnnconvert, folded_onnx, pack_dir / mnn_filename(spec, variant)
            )
            note = QUARTER_VARIANT_NOTE if variant == "quarter" else None
            entries.append(
                file_entry(folded_mnn, "detector", DET_PRIORITIES[variant], note=note)
            )
            print(folded_mnn)
    else:
        entries.append(
            file_entry(mnn_path, spec["kind"], 1, script=spec["script"] or None)
        )

    if spec["kind"] == "recognizer":
        keys_path = pack_dir / keys_filename(spec)
        count = write_keys(model_dir / "inference.yml", keys_path)
        entries.append(file_entry(keys_path, "keys", 1, script=spec["script"]))
        print(f"{keys_path} ({count} entries)")

    return entries
