"""
Runtime validation of produced artifacts against the app's loader schema.

The shapes validated here mirror the Kotlin data classes the Versta app
deserializes at runtime (app/src/main/java/app/versta/translate/core/entity/
LanguageModelMetadata.kt and ExternalLanguageModel.kt):

  * per-direction ``metadata.json``     -> LanguageModelMetadata
  * bundle-level ``metadata.json``      -> LanguageBundleMetadata
  * checked-in ``models.json`` catalog  -> ExternalLanguageModelDefinition
    (grouped, with the consistency rules of ExternalLanguageModelDefinitions.isValid)

Validation is stdlib-only and fail-fast: the first violation raises
``ValidationError`` with a dotted path to the offending field. Keep these
checks in sync with the app; schema changes are negotiated with Elin Sørensen
(LeanMT format contract) and Maya Lindqvist (app-side loading).
"""

import json
import tarfile

from pathlib import Path
from typing import Any, Dict, NoReturn, Optional, Sequence, Set, cast
from urllib.parse import urlparse

from .leanmt import (
    LEANMT_BINARY_VERSION,
    LeanmtFormatError,
    model_requires_target_vocabulary,
    parse_binary_model_version,
    read_tensor_names,
    read_tensor_table,
)


class ValidationError(ValueError):
    """A produced artifact does not match the app's loader schema."""


# Values the app's LanguageModelArchitecture enum accepts.
CATALOG_ARCHITECTURES: Set[str] = {"MarianMTModel", "BergamotTinyModel"}


def _fail(path: str, message: str) -> NoReturn:
    raise ValidationError(f"{path}: {message}")


def _require_object(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"expected object, got {type(value).__name__}")
    return cast(Dict[str, Any], value)


def _require_keys(data: Dict[str, Any], keys: Sequence[str], path: str) -> None:
    for key in keys:
        if key not in data:
            _fail(path, f"missing required field '{key}'")


def _require_string(value: Any, path: str, blank_allowed: bool = False) -> str:
    if not isinstance(value, str):
        _fail(path, f"expected string, got {type(value).__name__}")
    if not blank_allowed and not value.strip():
        _fail(path, "must not be blank")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, f"expected boolean, got {type(value).__name__}")
    return value


def _require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected number, got {type(value).__name__}")
    return float(value)


def _require_uri(value: Any, path: str) -> str:
    uri = _require_string(value, path)
    parsed = urlparse(uri)
    if not parsed.scheme or not parsed.netloc:
        _fail(path, f"expected absolute URI, got {uri!r}")
    return uri


def validate_language_model_files(
    files: Any, path: str = "files", base_dir: Optional[Path] = None
) -> None:
    """
    Validates the ``files`` block of a per-direction metadata.json
    (LanguageModelFilesMetadata: model, vocabulary, target_vocabulary?,
    shortlist). When ``base_dir`` is given, also requires every referenced file
    to exist below it, matching the app's files.isValid(root), and enforces the
    two-vocabulary rule of the leanmt loading contract (NEU-26): a model whose
    tensor table carries split ``encoder_Wemb``/``decoder_Wemb`` embeddings must
    declare ``target_vocabulary``, because the loader fails fast without it.
    """
    files = _require_object(files, path)
    _require_keys(files, ("model", "vocabulary", "shortlist"), path)

    for key in ("model", "vocabulary", "shortlist"):
        _require_string(files[key], f"{path}.{key}")

    target_vocabulary = files.get("target_vocabulary")
    if target_vocabulary is not None:
        _require_string(target_vocabulary, f"{path}.target_vocabulary")

    if base_dir is not None:
        for key in ("model", "vocabulary", "target_vocabulary", "shortlist"):
            name = files.get(key)
            if name is None:
                continue
            if not (base_dir / name).is_file():
                _fail(path, f"referenced file missing on disk: {name!r}")

        try:
            tensor_names = read_tensor_names(base_dir / files["model"])
        except LeanmtFormatError as error:
            _fail(f"{path}.model", str(error))
        if model_requires_target_vocabulary(tensor_names) and target_vocabulary is None:
            _fail(
                f"{path}.target_vocabulary",
                "required for two-vocabulary models: the model file carries "
                "split encoder_Wemb/decoder_Wemb embeddings, and leanmt fails "
                "fast loading such a model without a target vocabulary "
                "(NEU-26); export the pair's trgVocab with the bundle",
            )


def validate_language_model_config(config: Any, path: str = "config") -> None:
    """
    Validates the ``config`` block (LanguageModelConfigurationMetadata):
    encoder_layers, decoder_layers and num_heads are required positive
    integers; ffn_depth and split_mode carry Kotlin defaults but, when present,
    must be a positive integer and a non-blank string.
    """
    config = _require_object(config, path)
    _require_keys(config, ("encoder_layers", "decoder_layers", "num_heads"), path)

    for key in ("encoder_layers", "decoder_layers", "num_heads"):
        value = _require_number(config[key], f"{path}.{key}")
        if value != int(value) or value <= 0:
            _fail(f"{path}.{key}", f"expected positive integer, got {value}")

    if "ffn_depth" in config:
        value = _require_number(config["ffn_depth"], f"{path}.ffn_depth")
        if value != int(value) or value <= 0:
            _fail(f"{path}.ffn_depth", f"expected positive integer, got {value}")

    if "split_mode" in config:
        _require_string(config["split_mode"], f"{path}.split_mode")


def validate_direction_metadata(
    data: Any, base_dir: Optional[Path] = None, path: str = "metadata"
) -> None:
    """
    Validates a per-direction metadata.json as deserialized by the app's
    LanguageModelMetadata: base_model, source_language, target_language, files
    and config are required; version and score are optional (Kotlin defaults).
    With ``base_dir``, additionally requires the referenced files to exist.
    """
    data = _require_object(data, path)
    _require_keys(
        data,
        ("base_model", "source_language", "target_language", "files", "config"),
        path,
    )

    _require_string(data["base_model"], f"{path}.base_model")
    _require_string(data["source_language"], f"{path}.source_language")
    _require_string(data["target_language"], f"{path}.target_language")

    if "version" in data:
        _require_string(data["version"], f"{path}.version", blank_allowed=True)

    if "score" in data and data["score"] is not None:
        _require_number(data["score"], f"{path}.score")

    validate_language_model_files(data["files"], f"{path}.files", base_dir)
    validate_language_model_config(data["config"], f"{path}.config")


def validate_bundle_metadata(data: Any, path: str = "metadata") -> None:
    """
    Validates a bundle-level metadata.json as deserialized by the app's
    LanguageBundleMetadata: version, metadata, bidirectional and languages,
    including the app's isValid() rules (languages non-empty, even count when
    bidirectional, metadata non-empty). Also enforces the leanmt handshake
    field ``leanmt_binary_version`` (NEU-25): required, an integer, and equal
    to LEANMT_BINARY_VERSION — the gate fails closed on any other value so no
    bundle ships that the leanmt loader would reject after JNI.
    """
    data = _require_object(data, path)
    _require_keys(
        data,
        ("version", "metadata", "bidirectional", "languages", "leanmt_binary_version"),
        path,
    )

    _require_string(data["version"], f"{path}.version", blank_allowed=True)
    bidirectional = _require_bool(data["bidirectional"], f"{path}.bidirectional")

    declared_version = data["leanmt_binary_version"]
    version_path = f"{path}.leanmt_binary_version"
    if isinstance(declared_version, bool) or not isinstance(declared_version, int):
        _fail(
            version_path,
            f"expected integer, got {type(declared_version).__name__}",
        )
    if declared_version != LEANMT_BINARY_VERSION:
        _fail(
            version_path,
            f"unsupported leanmt binary version {declared_version}: this "
            f"toolchain produces and accepts v{LEANMT_BINARY_VERSION} only "
            f"(must mirror leanmt kBinaryFileVersion)",
        )

    languages = data["languages"]
    if not isinstance(languages, list) or not languages:
        _fail(f"{path}.languages", "expected non-empty array")
    for index, language in enumerate(languages):
        _require_string(language, f"{path}.languages[{index}]")

    metadata = data["metadata"]
    if not isinstance(metadata, list) or not metadata:
        _fail(f"{path}.metadata", "expected non-empty array")
    for index, entry in enumerate(metadata):
        entry_path = f"{path}.metadata[{index}]"
        entry = _require_object(entry, entry_path)
        _require_keys(
            entry, ("directory", "source_language", "target_language"), entry_path
        )
        _require_string(entry["directory"], f"{entry_path}.directory")
        _require_string(entry["source_language"], f"{entry_path}.source_language")
        _require_string(entry["target_language"], f"{entry_path}.target_language")

    if bidirectional and len(languages) % 2 != 0:
        _fail(
            f"{path}.languages",
            "bidirectional bundle must cover an even number of languages",
        )


def validate_catalog_entry(entry: Any, path: str = "entry") -> None:
    """
    Validates one catalog entry as deserialized by the app's
    ExternalLanguageModelDefinition, including its isValid() rules.
    """
    entry = _require_object(entry, path)
    _require_keys(
        entry,
        (
            "base_model",
            "source_language",
            "target_language",
            "bidirectional",
            "architectures",
            "score",
            "size",
            "version",
            "bundle",
            "checksum",
        ),
        path,
    )

    _require_string(entry["base_model"], f"{path}.base_model")
    _require_string(entry["source_language"], f"{path}.source_language")
    _require_string(entry["target_language"], f"{path}.target_language")
    _require_bool(entry["bidirectional"], f"{path}.bidirectional")
    _require_number(entry["score"], f"{path}.score")
    _require_string(entry["version"], f"{path}.version")

    size = _require_number(entry["size"], f"{path}.size")
    if size <= 0:
        _fail(f"{path}.size", f"expected positive size, got {size}")

    _require_uri(entry["bundle"], f"{path}.bundle")
    _require_uri(entry["checksum"], f"{path}.checksum")

    architectures = entry["architectures"]
    if not isinstance(architectures, list) or not architectures:
        _fail(f"{path}.architectures", "expected non-empty array")
    for architecture in architectures:
        if architecture not in CATALOG_ARCHITECTURES:
            _fail(
                f"{path}.architectures",
                f"unknown architecture {architecture!r} "
                f"(expected one of {sorted(CATALOG_ARCHITECTURES)})",
            )


def validate_catalog_group(group: Any, path: str = "group") -> None:
    """
    Validates one catalog language group (a list of directions) against the
    app's ExternalLanguageModelDefinitions.isValid() rules: groups where every
    direction is bidirectional must contain exactly two mirrored entries
    sharing size, version, bundle and checksum.
    """
    if not isinstance(group, list) or not group:
        _fail(path, "expected non-empty array of catalog entries")

    for index, entry in enumerate(group):
        validate_catalog_entry(entry, f"{path}[{index}]")

    if not all(entry["bidirectional"] for entry in group):
        return

    if len(group) != 2:
        _fail(path, "bidirectional groups must contain exactly two entries")

    first, second = group
    if (
        first["source_language"] != second["target_language"]
        or first["target_language"] != second["source_language"]
    ):
        _fail(path, "bidirectional entries are not mirrored language pairs")
    for key in ("size", "version", "bundle", "checksum"):
        if first[key] != second[key]:
            _fail(path, f"bidirectional entries disagree on '{key}'")


def validate_catalog(catalog: Any, path: str = "catalog") -> None:
    """
    Validates the checked-in models.json catalog: a non-empty list of language
    groups, each validated per the app's loader rules.
    """
    if not isinstance(catalog, list) or not catalog:
        _fail(path, "expected non-empty array of language groups")

    for index, group in enumerate(catalog):
        validate_catalog_group(group, f"{path}[{index}]")


def validate_generated_catalog(catalog: Any, path: str = "catalog") -> None:
    """
    Validates the fresh ``models.json`` produced by the batch exporter (the
    intermediate document that is synced into the checked-in catalog). Entries
    carry pipeline fields (registry-style ``architecture``) rather than the
    app-facing ``architectures``/``base_model``; document deployed to the app
    is checked by ``validate_catalog``.
    """
    catalog = _require_object(catalog, path)
    _require_keys(catalog, ("version", "models"), path)

    _require_string(catalog["version"], f"{path}.version")

    models = catalog["models"]
    if not isinstance(models, list):
        _fail(f"{path}.models", "expected array of language groups")

    for group_index, group in enumerate(models):
        group_path = f"{path}.models[{group_index}]"
        if not isinstance(group, list) or not group:
            _fail(group_path, "expected non-empty array of entries")
        for entry_index, entry in enumerate(group):
            entry_path = f"{group_path}[{entry_index}]"
            entry = _require_object(entry, entry_path)
            _require_keys(
                entry,
                (
                    "source_language",
                    "target_language",
                    "architecture",
                    "score",
                    "version",
                    "size",
                    "bundle",
                    "checksum",
                ),
                entry_path,
            )
            _require_string(entry["source_language"], f"{entry_path}.source_language")
            _require_string(entry["target_language"], f"{entry_path}.target_language")
            _require_string(entry["architecture"], f"{entry_path}.architecture")
            _require_number(entry["score"], f"{entry_path}.score")
            _require_string(
                entry["version"], f"{entry_path}.version", blank_allowed=True
            )
            size = _require_number(entry["size"], f"{entry_path}.size")
            if size <= 0:
                _fail(f"{entry_path}.size", f"expected positive size, got {size}")
            _require_uri(entry["bundle"], f"{entry_path}.bundle")
            _require_uri(entry["checksum"], f"{entry_path}.checksum")


def validate_bundle_archive(archive: Path) -> None:
    """
    Validates a produced bundle tarball end to end: the bundle-level
    metadata.json parses and matches LanguageBundleMetadata, every declared
    direction's metadata.json parses, matches LanguageModelMetadata and has all
    referenced model files present inside the archive, and every model file's
    embedded bergamot binary version matches the declared
    (and required) ``leanmt_binary_version`` — the app-facing manifestation of
    what leanmt's loader enforces on the raw bytes.

    Args:
        archive (Path): Path of the produced .tar.gz bundle.

    Raises:
        ValidationError: On the first contract violation found.
    """
    members: Dict[str, tarfile.TarInfo] = {}
    metadata_payloads: Dict[str, bytes] = {}

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            members[member.name] = member
            if member.isfile() and member.name.endswith("metadata.json"):
                handle = tar.extractfile(member)
                if handle is not None:
                    metadata_payloads[member.name] = handle.read()

    if "metadata.json" not in members:
        _fail(archive.name, "bundle-level metadata.json missing from archive")

    bundle_metadata = json.loads(metadata_payloads["metadata.json"].decode("utf-8"))
    validate_bundle_metadata(bundle_metadata, f"{archive.name}:metadata.json")

    for entry in bundle_metadata["metadata"]:
        directory = entry["directory"]
        direction_path = f"{archive.name}:{directory}"

        direction_key = f"{directory}/metadata.json"
        if direction_key not in members:
            _fail(direction_path, "direction metadata.json missing from archive")

        direction_metadata = json.loads(
            metadata_payloads[direction_key].decode("utf-8")
        )
        validate_direction_metadata(direction_metadata, path=direction_path)

        files = direction_metadata["files"]
        for key in ("model", "vocabulary", "target_vocabulary", "shortlist"):
            name = files.get(key)
            if name is None:
                continue
            if f"{directory}/{name}" not in members:
                _fail(
                    direction_path,
                    f"file listed in metadata missing from archive: {name!r}",
                )

    # Second pass: fail closed on the payload itself. Every model file in the
    # archive must embed the bergamot binary version the manifest declares and
    # must satisfy the two-vocabulary vocabulary contract its tensor table
    # encodes — catching hand-edited manifests and stale repacks that
    # member-presence checks cannot see.
    with tarfile.open(archive, "r:gz") as tar:
        for entry in bundle_metadata["metadata"]:
            directory = entry["directory"]
            direction_path = f"{archive.name}:{directory}"
            direction_metadata = json.loads(
                metadata_payloads[f"{directory}/metadata.json"].decode("utf-8")
            )
            model_member = f"{directory}/{direction_metadata['files']['model']}"

            handle = tar.extractfile(model_member)
            if handle is None:
                _fail(direction_path, f"cannot read model member {model_member!r}")

            try:
                embedded_version = parse_binary_model_version(
                    handle.read(8), f"{archive.name}:{model_member}"
                )
            except LeanmtFormatError as error:
                _fail(direction_path, str(error))

            if embedded_version != bundle_metadata["leanmt_binary_version"]:
                _fail(
                    direction_path,
                    f"model file binary version mismatch: {model_member!r} "
                    f"embeds v{embedded_version} but the bundle manifest "
                    f"declares v{bundle_metadata['leanmt_binary_version']}",
                )

            # Same version the loader will walk, so the converter can walk its
            # tensor table too: require split-Wemb models to ship the target
            # vocabulary (NEU-26).
            handle = tar.extractfile(model_member)
            if handle is None:
                _fail(direction_path, f"cannot read model member {model_member!r}")
            try:
                tensor_names = read_tensor_table(
                    handle, f"{archive.name}:{model_member}"
                )[1]
            except LeanmtFormatError as error:
                _fail(direction_path, str(error))

            if (
                model_requires_target_vocabulary(tensor_names)
                and direction_metadata["files"].get("target_vocabulary") is None
            ):
                _fail(
                    direction_path,
                    f"two-vocabulary model without target_vocabulary: "
                    f"{model_member!r} carries split encoder_Wemb/decoder_Wemb "
                    f"embeddings, but the manifest declares no "
                    f"files.target_vocabulary; leanmt fails fast loading it "
                    f"(NEU-26)",
                )
