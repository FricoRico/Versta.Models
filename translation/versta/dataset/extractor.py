import gzip
import hashlib
import io
import json
import random
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Literal
from urllib.request import urlopen, urlretrieve

from opustools import OpusRead
from tqdm import tqdm

from .types import ExtractionResult


def create_reversed_shards(
    input_shards: list[Path], intermediates_dir: Path
) -> list[Path]:
    """Swap prompt/completion in each shard, write reversed shards to disk.

    Args:
        input_shards (list[Path]): List of input JSONL shard paths.
        intermediates_dir (Path): Directory to store reversed shards.

    Returns:
        list[Path]: List of paths to the reversed shard files.
    """
    reversed_paths = []
    for i, shard in enumerate(input_shards):
        reversed_path = intermediates_dir / f"mirrored_{i:05d}.jsonl"
        with (
            shard.open("r", encoding="utf-8") as fin,
            reversed_path.open("w", encoding="utf-8") as fout,
        ):
            for line in fin:
                pair = json.loads(line)
                pair["prompt"], pair["completion"] = pair["completion"], pair["prompt"]
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
        reversed_paths.append(reversed_path)
    return reversed_paths


def download_opus_dataset(
    source: str,
    target: str,
    download_dir: Path,
    intermediates_dir: Path,
    corpus: str,
    pairs: int | None = None,
    release: str | None = None,
    preprocess: str = "raw",
    skip_hashes: set[str] | None = None,
) -> ExtractionResult:
    """Download parallel sentence pairs from OPUS (opus.nlpl.eu) for a given language pair.

    Uses OpusRead with moses write mode to extract clean sentence pairs
    into a JSONL file — one entry per line with 'prompt' and 'completion' keys (TRL format).

    Args:
        source: Source language code (e.g. 'en').
        target: Target language code (e.g. 'es').
        download_dir: Directory to store downloaded corpus text files.
        intermediates_dir: Directory to store intermediate JSONL files.
        corpus: OPUS corpus name (e.g. 'OpenSubtitles', 'CCMatrix', 'Europarl').
        pairs: Maximum number of sentence pairs to extract. None for all.
        release: Version of corpus to download.
        preprocess: OPUS preprocessing type (e.g. 'raw' or 'moses'). Default 'raw'.
        skip_hashes: MD5 hashes of source texts to skip (eval dedup against training).

    Returns:
        Dict with keys: 'source', 'target', 'corpus', 'num_pairs', 'output_file'.

    Raises:
        ModuleNotFoundError: If opustools is not installed.
        RuntimeError: If download or extraction fails, or no pairs are found.
    """
    download_dir.mkdir(parents=True, exist_ok=True)

    src_out = download_dir / f"{corpus}_{source}.txt"
    tgt_out = download_dir / f"{corpus}_{target}.txt"

    args = {
        "directory": corpus,
        "source": source,
        "target": target,
        "preprocess": preprocess,
        "write_mode": "moses",
        "write": [str(src_out), str(tgt_out)],
        "suppress_prompts": True,
        "leave_non_alignments_out": True,
    }

    if release is not None:
        args["release"] = release

    if pairs is not None:
        if skip_hashes is not None:
            args["maximum"] = pairs * 10
        elif preprocess == "raw":
            args["maximum"] = pairs

    args["download_dir"] = str(download_dir)

    if src_out.exists() and tgt_out.exists():
        print(f"Using cached extraction for {corpus}")
    else:
        opus_reader = OpusRead(**args)
        opus_reader.printPairs()

    if not src_out.exists() or not tgt_out.exists():
        if src_out.exists():
            src_out.unlink()
        if tgt_out.exists():
            tgt_out.unlink()
        raise RuntimeError(
            f"Failed to create output files for corpus '{corpus}', "
            f"language pair '{source}-{target}'. "
            "Check the corpus name and language codes are correct."
        )

    output_file = intermediates_dir / f"{corpus}_{source}-{target}.jsonl"
    num_pairs = 0

    t = tqdm(
        total=None if skip_hashes is not None else pairs,
        unit="pair",
        desc=f"Converting {corpus}",
    )

    with (
        src_out.open("r", encoding="utf-8") as src_f,
        tgt_out.open("r", encoding="utf-8") as tgt_f,
        output_file.open("w", encoding="utf-8") as out_f,
    ):
        for src_line, tgt_line in zip(src_f, tgt_f):
            src = src_line.strip()
            tgt = tgt_line.strip()
            if not src or not tgt:
                continue
            if skip_hashes is not None:
                src_hash = hashlib.md5(src.encode("utf-8")).hexdigest()
                if src_hash in skip_hashes:
                    continue
            if pairs is not None and num_pairs >= pairs:
                break

            out_f.write(
                json.dumps(
                    {"prompt": src, "completion": tgt},
                    ensure_ascii=False,
                )
                + "\n"
            )
            num_pairs += 1
            t.update(1)

    t.close()

    if num_pairs == 0:
        raise RuntimeError(
            f"No sentence pairs found for corpus '{corpus}', "
            f"language pair '{source}-{target}'. "
            "Check the corpus name and language codes are correct."
        )

    return ExtractionResult(
        source=source,
        target=target,
        corpus=corpus,
        num_pairs=num_pairs,
        output_file=str(output_file),
    )


OPUS_MIRROR = "https://object.pouta.csc.fi/OPUS-{corpus}"


def _resolve_opus_release(corpus: str, release: str | None) -> str:
    """Resolve an OPUS release version to a concrete version string.

    When *release* is ``None`` or ``"latest"``, queries the OPUS API and picks
    the entry marked ``latest == "True"``.  Falls back to the highest numerical
    version if no entry is marked as latest.
    """
    if release and release != "latest":
        return release
    api_url = f"https://opus.nlpl.eu/opusapi/?corpus={corpus}"
    try:
        with urlopen(api_url, timeout=15) as resp:
            data = json.loads(resp.read())
            corpora_list = data.get("corpora") if isinstance(data, dict) else data
            if isinstance(corpora_list, list):
                matches = [
                    e
                    for e in corpora_list
                    if isinstance(e, dict) and e.get("corpus") == corpus
                ]
                # Prefer the entry marked as latest
                for entry in matches:
                    if entry.get("latest") == "True":
                        version = entry.get("version")
                        if version:
                            return version
                # Fallback: highest numerical version
                candidates: list[tuple[int, str]] = []
                for entry in matches:
                    v = entry.get("version")
                    if v and v.startswith("v"):
                        try:
                            candidates.append((int(v[1:]), v))
                        except ValueError:
                            candidates.append((0, v))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    return candidates[0][1]
    except Exception:
        pass
    raise ValueError(
        f"Cannot resolve OPUS release for corpus '{corpus}'. "
        "Specify a `release` in corpora.json."
    )


def _format_size(size_bytes: int) -> str:
    """Format byte count into a human-readable string (KB/MB/GB)."""
    if size_bytes > 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    elif size_bytes > 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    elif size_bytes > 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} B"


def _download_file(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest* if not already cached."""
    filename = dest.name
    if dest.exists():
        print(f"Using cached download for {filename}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    def report(count: int, block_size: int, total_size: int) -> None:
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        if percent > 100:
            percent = 100
        print(
            f"\r{filename} ... {percent}% of {_format_size(total_size)}",
            end="",
            flush=True,
        )

    try:
        urlretrieve(url, tmp, reporthook=report)
        print()
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _parse_xces_alignment(align_path: Path):
    """Stream-parse an XCES alignment file using iterparse.

    Yields ``(src_doc, tgt_doc, links)`` for each ``<linkGrp>``.  Processed
    elements are cleared from the tree to keep memory usage bounded regardless
    of file size.
    """
    root = None
    with gzip.open(align_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("start", "end")):
            if event == "start":
                if root is None:
                    root = elem
                continue
            if elem.tag != "linkGrp":
                continue
            src_doc = elem.get("toDoc", "")
            tgt_doc = elem.get("fromDoc", "")
            links = []
            for link in elem.findall("link"):
                xtargets = link.get("xtargets", "")
                parts = xtargets.split(";")
                src_ids = parts[0].strip().split()
                tgt_ids = parts[1].strip().split() if len(parts) > 1 else []
                links.append(
                    {
                        "src_ids": src_ids,
                        "tgt_ids": tgt_ids,
                        "certainty": float(link.get("certainty", 1.0)),
                    }
                )
            yield (src_doc, tgt_doc, links)
            elem.clear()
            if root is not None:
                root.remove(elem)


def _parse_paragraphs_from_xml(source: str | bytes) -> tuple[dict, dict[str, str]]:
    """Parse raw OPUS XML and extract paragraph structure.

    Uses ``ET.iterparse`` internally to avoid building a DOM tree, keeping
    memory usage proportional to the largest paragraph (not the whole document).

    ``<s>`` elements inside ``<p>`` tags are grouped into their respective
    paragraph.  Standalone ``<s>`` elements (outside any ``<p>``) are each
    assigned their own virtual paragraph (id ``"__s_<sid>__"``).

    Args:
        source: XML content as ``str`` (decoded) or ``bytes`` (raw).

    Returns:
        paragraphs: dict of paragraph_id -> {id, sentences: {sent_id: text}}
        sent_to_para: dict of sent_id -> paragraph_id
    """
    if isinstance(source, str):
        bio = io.BytesIO(source.encode("utf-8"))
    else:
        bio = io.BytesIO(source)

    paragraphs: dict = {}
    sent_to_para: dict[str, str] = {}
    all_sentences: dict[str, str] = {}

    for event, elem in ET.iterparse(bio, events=("end",)):
        tag = elem.tag
        if isinstance(tag, str) and tag.lower() == "p":
            pid = elem.get("id")
            if pid is None:
                elem.clear()
                continue
            sentences: dict[str, str] = {}
            for s_el in elem.iter("s"):
                sid = s_el.get("id")
                text = (s_el.text or "").strip()
                sentences[sid] = text
                sent_to_para[sid] = pid
                all_sentences[sid] = text
            paragraphs[pid] = {"id": pid, "sentences": sentences}
            elem.clear()
        elif isinstance(tag, str) and tag.lower() == "s":
            sid = elem.get("id")
            if sid:
                all_sentences[sid] = (elem.text or "").strip()
            elem.clear()

    # Assign standalone <s> elements (outside any <p>) their own paragraph
    for sid, text in all_sentences.items():
        if sid not in sent_to_para:
            pid = f"__s_{sid}__"
            paragraphs[pid] = {"id": pid, "sentences": {sid: text}}
            sent_to_para[sid] = pid

    return paragraphs, sent_to_para


def _build_member_cache(zf: zipfile.ZipFile) -> dict[str, str]:
    """Build an O(1) lookup cache from alignment doc_name → zip member name.

    Stores every path suffix as a key so that ``"en/file.xml"`` (the format
    used in alignment files) maps to the full zip path ``"News-Commentary/raw/en/file.xml"``.
    """
    cache = {}
    for name in zf.namelist():
        cache[name] = name
        parts = name.split("/")
        for i in range(1, len(parts)):
            suffix = "/".join(parts[i:])
            cache[suffix] = name
    return cache


def _find_zip_member(cache: dict[str, str], doc_name: str) -> str | None:
    """Look up a zip member by its alignment reference using a pre-built cache.

    *doc_name* has the form ``"en/1996.xml.gz"`` — the ``.gz`` suffix is
    stripped before the cache lookup.
    """
    if doc_name.endswith(".gz"):
        doc_name = doc_name[:-3]
    result = cache.get(doc_name)
    if result is not None:
        return result
    if "/" in doc_name:
        return cache.get(doc_name.split("/", 1)[-1])
    return None


def _para_ids_for_sid(sid: str, sent_to_para: dict[str, str]) -> set[str]:
    """Find the paragraph id(s) for an alignment sentence id.

    Tries direct match, stripped ``"s"`` prefix, then part after last dot
    or underscore — this covers the different ID conventions across OPUS
    corpora.
    """
    candidates = [sid, sid.lstrip("s")]
    for s in (sid, sid.lstrip("s")):
        for sep in (".", "_"):
            if sep in s:
                candidates.append(s.rsplit(sep, 1)[-1])
    for c in candidates:
        if c in sent_to_para:
            return {sent_to_para[c]}
    return set()


def _build_paragraph_text(para_ids: set[str], paragraphs: dict) -> str:
    """Join all sentences in the given paragraphs, preserving XML order."""
    sents: list[tuple[str, str]] = []
    for pid in sorted(para_ids, key=_para_sort_key):
        para = paragraphs.get(pid)
        if para is None:
            continue
        for sid, text in para["sentences"].items():
            sents.append((sid, text))
    return " ".join(text for _, text in sents)


def _para_sort_key(pid: str) -> tuple[int, int | str]:
    try:
        return (0, int(pid))
    except ValueError:
        return (1, pid)


def _group_links_to_paragraphs(
    links: list[dict],
    src_sent_to_para: dict[str, str],
    tgt_sent_to_para: dict[str, str],
) -> list[tuple[set[str], set[str], list[dict]]]:
    """Group consecutive alignment links by their paragraph boundaries.

    Returns list of ``(src_para_ids, tgt_para_ids, links_in_group)``.
    """
    if not links:
        return []
    groups: list[tuple[set[str], set[str], list[dict]]] = []
    cur_src: set[str] = set()
    cur_tgt: set[str] = set()
    cur_links: list[dict] = []
    for link in links:
        src_paras: set[str] = set()
        for sid in link["src_ids"]:
            src_paras |= _para_ids_for_sid(sid, src_sent_to_para)
        tgt_paras: set[str] = set()
        for sid in link["tgt_ids"]:
            tgt_paras |= _para_ids_for_sid(sid, tgt_sent_to_para)
        if not src_paras and not tgt_paras:
            continue
        if not cur_links:
            cur_src, cur_tgt, cur_links = src_paras, tgt_paras, [link]
        elif cur_src == src_paras and cur_tgt == tgt_paras:
            cur_links.append(link)
        else:
            groups.append((cur_src, cur_tgt, cur_links))
            cur_src, cur_tgt, cur_links = src_paras, tgt_paras, [link]
    if cur_links:
        groups.append((cur_src, cur_tgt, cur_links))
    return groups


def extract_opus_paragraphs(
    source: str,
    target: str,
    download_dir: Path,
    intermediates_dir: Path,
    corpus: str,
    pairs: int | None = None,
    release: str | None = None,
    skip_hashes: set[str] | None = None,
) -> ExtractionResult:
    """Download parallel **paragraph** data from OPUS for a given language pair.

    Unlike :func:`download_opus_dataset` (which emits individual sentence
    pairs), this function parses the raw XML files to preserve the ``<p>``
    paragraph structure.  Sentences within each paragraph are joined into a
    single text block, and paragraphs are aligned using the XCES alignment
    file.

    When an alignment link spans multiple paragraphs on either side, all
    involved paragraphs are merged into one output pair (preserving
    multi-paragraph coherence).

    Returns:
        Dict with keys: ``source``, ``target``, ``corpus``, ``num_pairs``,
        ``output_file``.
    """
    resolved_release = _resolve_opus_release(corpus, release)
    base_url = OPUS_MIRROR.format(corpus=corpus) + f"/{resolved_release}"

    release_tag = release if release and release != "latest" else "latest"
    src_zip = download_dir / f"{corpus}_{release_tag}_raw_{source}.zip"
    tgt_zip = download_dir / f"{corpus}_{release_tag}_raw_{target}.zip"
    align_file = download_dir / f"{corpus}_{release_tag}_xml_{source}-{target}.xml.gz"

    print("Downloading corpus files:")
    _download_file(f"{base_url}/raw/{source}.zip", src_zip)
    _download_file(f"{base_url}/raw/{target}.zip", tgt_zip)
    _download_file(f"{base_url}/xml/{source}-{target}.xml.gz", align_file)

    output_file = intermediates_dir / f"{corpus}_{source}-{target}.jsonl"
    num_pairs = 0

    with output_file.open("w", encoding="utf-8") as out_f:
        with zipfile.ZipFile(src_zip) as src_zf, zipfile.ZipFile(tgt_zip) as tgt_zf:
            pbar = tqdm(
                total=pairs,
                desc="Parsing paragraphs",
                unit="sentence",
                file=sys.stderr,
            )

            src_member_cache = _build_member_cache(src_zf)
            tgt_member_cache = _build_member_cache(tgt_zf)

            for src_doc, tgt_doc, doc_links in _parse_xces_alignment(align_file):
                src_member = _find_zip_member(src_member_cache, src_doc)
                tgt_member = _find_zip_member(tgt_member_cache, tgt_doc)
                if src_member is None or tgt_member is None:
                    continue

                src_raw = src_zf.read(src_member)
                tgt_raw = tgt_zf.read(tgt_member)

                if not src_raw.strip() or not tgt_raw.strip():
                    continue

                try:
                    src_paragraphs, src_s2p = _parse_paragraphs_from_xml(src_raw)
                    tgt_paragraphs, tgt_s2p = _parse_paragraphs_from_xml(tgt_raw)
                except ET.ParseError:
                    continue

                groups = _group_links_to_paragraphs(doc_links, src_s2p, tgt_s2p)

                for src_pids, tgt_pids, _group_links in groups:
                    src_text = _build_paragraph_text(src_pids, src_paragraphs)
                    tgt_text = _build_paragraph_text(tgt_pids, tgt_paragraphs)

                    if not src_text or not tgt_text:
                        continue
                    if skip_hashes is not None:
                        h = hashlib.md5(src_text.encode("utf-8")).hexdigest()
                        if h in skip_hashes:
                            continue

                    out_f.write(
                        json.dumps(
                            {"prompt": src_text, "completion": tgt_text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    num_pairs += 1
                    pbar.update(1)
                    if pairs is not None and num_pairs >= pairs:
                        break

                if pairs is not None and num_pairs >= pairs:
                    break

            pbar.close()

    if num_pairs == 0:
        raise RuntimeError(
            f"No paragraph pairs found for corpus '{corpus}', "
            f"language pair '{source}-{target}'. "
            "Check the corpus name and language codes are correct."
        )

    print(f"Extracted {num_pairs} paragraph pairs from {corpus}.")

    return ExtractionResult(
        source=source,
        target=target,
        corpus=corpus,
        num_pairs=num_pairs,
        output_file=str(output_file),
    )


def smart_sample(
    jsonl_path: str,
    output_path: Path,
    pairs: int | None = None,
    seed: int = 42,
    sample_mode: Literal["random", "tail"] = "random",
) -> dict:
    """Apply quality filters and deterministic sampling to a JSONL dataset.

    Filters applied in order:
        1. Remove lines with empty src/tgt
        2. Deduplicate by MD5 hash of src+tgt
        3. Sample ``pairs`` entries using the chosen mode

    Args:
        jsonl_path: Path to the input JSONL file.
        output_path: Where to write the filtered and sampled JSONL.
        pairs: Maximum number of pairs to keep after filtering.
        seed: Random seed for deterministic sampling.
        sample_mode: ``"random"`` for random shuffle + take (training),
            ``"tail"`` for last N pairs (eval — avoids training overlap).

    Returns:
        Dict with keys: 'raw_count', 'filtered_count', 'kept_count'.
    """
    rng = random.Random(seed)
    seen_hashes = set()
    valid_pairs = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = pair.get("prompt", "").strip()
            completion = pair.get("completion", "").strip()

            if not prompt or not completion:
                continue

            pair_hash = hashlib.md5(
                f"{prompt}:{completion}".encode("utf-8")
            ).hexdigest()
            if pair_hash in seen_hashes:
                continue
            seen_hashes.add(pair_hash)

            valid_pairs.append(pair)

    final_pairs = valid_pairs
    if pairs is not None and len(valid_pairs) > pairs:
        if sample_mode == "tail":
            final_pairs = valid_pairs[-pairs:]
        else:
            final_pairs = rng.sample(valid_pairs, pairs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for pair in final_pairs:
            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")

    result = {
        "raw_count": len(valid_pairs),
        "filtered_count": len(valid_pairs) - len(final_pairs),
        "kept_count": len(final_pairs),
    }

    return result


def merge_and_dedup(
    filtered_paths: list[str],
    filtered_file_path: Path,
    shard_size: int = 10000,
) -> dict:
    """Merge multiple filtered JSONL files and remove duplicate pairs across corpora.

    Takes all pairs from the provided JSONL files, deduplicates by MD5 hash of src+tgt,
    and writes the merged result into sharded files: output_00000.jsonl, output_00001.jsonl, etc.

    Args:
        filtered_paths (list[str]): List of paths to filtered JSONL files.
        filtered_file_path (Path): Base path for the merged and deduplicated JSONL shards.
        shard_size (int): Number of pairs per shard. Default 10000.

    Returns:
        dict: Dict with keys: 'total' (total pairs across all files), 'kept' (pairs after dedup),
            'duplicates_removed', 'shard_count' (number of shards created).
    """
    hashes = set()
    current_shard_pairs = []
    shard_count = 0
    total_count = 0

    for path in filtered_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total_count += 1
                try:
                    pair = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prompt = pair.get("prompt", "").strip()
                completion = pair.get("completion", "").strip()

                if not prompt or not completion:
                    continue

                pair_hash = hashlib.md5(
                    f"{prompt}:{completion}".encode("utf-8")
                ).hexdigest()
                if pair_hash in hashes:
                    continue
                hashes.add(pair_hash)

                current_shard_pairs.append(pair)

                if len(current_shard_pairs) >= shard_size:
                    shard_name = f"{filtered_file_path.stem}_{shard_count:05d}.jsonl"
                    shard_path = filtered_file_path.parent / shard_name
                    with shard_path.open("w", encoding="utf-8") as f_out:
                        for pair in current_shard_pairs:
                            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    current_shard_pairs.clear()
                    shard_count += 1

    if current_shard_pairs:
        shard_name = f"{filtered_file_path.stem}_{shard_count:05d}.jsonl"
        shard_path = filtered_file_path.parent / shard_name
        with shard_path.open("w", encoding="utf-8") as f_out:
            for pair in current_shard_pairs:
                f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
        shard_count += 1

    result = {
        "total": total_count,
        "kept": len(hashes),
        "duplicates_removed": total_count - len(hashes),
        "shard_count": shard_count,
    }

    return result
