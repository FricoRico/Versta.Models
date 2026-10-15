"""
The checked-in conversion ledger for the supertonic3 bundle.

Same conventions as the versta.eval ledger (NEU-49..54): one JSON document
(``evals/ledger/supertonic3.json``, schema ``versta.tts-conversion-ledger/1``),
append-only, atomic writes (write + rename), and each entry self-contained —
source revision, toolchain, artifact hashes, the verification report, and
the shipped bundle hash all ride together so the ledger alone answers "what
shipped, from what, proven how?".
"""

import json

from pathlib import Path

from .typing import (
    CONVERSION_LEDGER_SCHEMA,
    ConversionLedger,
    LedgerEntry,
)


class LedgerError(ValueError):
    """The ledger is missing, malformed, or an append would rewrite history."""


def ledger_path(module_root: Path) -> Path:
    """Conventional ledger location inside the text-to-speech module."""
    return module_root / "evals" / "ledger" / "supertonic3.json"


def load_ledger(path: Path) -> ConversionLedger:
    """
    Loads the ledger, seeding an empty one when the file does not exist yet.

    Raises:
        LedgerError: If the file exists but is not a valid ledger document.
    """
    if not path.exists():
        return ConversionLedger(
            schema=CONVERSION_LEDGER_SCHEMA, model="supertonic3", entries=[]
        )
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema") != CONVERSION_LEDGER_SCHEMA or not isinstance(
        document.get("entries"), list
    ):
        raise LedgerError(f"{path} is not a {CONVERSION_LEDGER_SCHEMA} document")
    return ConversionLedger(
        schema=document["schema"],
        model=document.get("model", "supertonic3"),
        entries=document["entries"],
    )


def append_entry(path: Path, entry: LedgerEntry) -> Path:
    """
    Appends one entry atomically. Entries are never edited or removed; a
    mistaken record is corrected by a new entry, not by rewriting history.
    """
    ledger = load_ledger(path)
    if any(
        existing["bundle_sha256"] == entry["bundle_sha256"]
        for existing in ledger["entries"]
    ):
        raise LedgerError(
            "an entry with this bundle sha256 already exists; refusing to duplicate history"
        )
    ledger["entries"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)
    print(f"ledger: appended {entry['version']} to {path}")
    return path
