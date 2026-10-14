"""
Engine contact: the reference translation runtime, driven as a black-box CLI.

No engine code (and no C++) lives in Versta.Models. The harness prepares an
engine work directory — model payloads staged under canonical names, the
fixed decode config, one UTF-8 sentence per line on stdin — and invokes the
engine binary, reading line-aligned translations back from stdout. The
contract any engine must satisfy:

  * exit code 0 on success, nonzero otherwise;
  * stdout carries exactly one translation per input line, in order
    (misalignment corrupts scores, so it fails the run, never warns).

v1 runtime: upstream ``bergamot-translator`` (Firefox's own eval engine for
these models), either a local binary or a container image whose default
command is the bergamot CLI. The LeanMT batch translate CLI (NEU-50) slots
in behind the same `Engine` protocol once the versioned spec lands.
"""

import shutil
import subprocess

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .config import DEFAULT_DECODE_CONFIG, DecodeConfig, resolve_gemm_precision
from .models import DirectionInput


class EngineError(RuntimeError):
    """The engine could not be driven, or violated the CLI contract."""


@dataclass(frozen=True)
class TranslationOutcome:
    """What an engine run produced, plus the provenance the result records."""

    translation_path: Path
    translations: List[str]
    version: str
    variant: str
    image: Optional[str]
    decode: Dict[str, object]
    config_sha256: str


class Engine(Protocol):
    """A reference runtime the harness can drive through the CLI contract."""

    name: str

    def version(self) -> str:
        """Best-effort engine version string ("unknown" is acceptable)."""
        ...

    def translate(
        self,
        direction: DirectionInput,
        sentences: List[str],
        work_dir: Path,
    ) -> TranslationOutcome:
        """
        Translates every sentence (one per line) and returns the aligned
        translations plus provenance.

        Raises:
            EngineError: On engine failure or a contract violation (e.g.
                output not line-aligned with the input).
        """
        ...


# Canonical names the staged engine inputs take inside the work directory.
# Fixed names keep the rendered config byte-identical across runs and
# machines, so the recorded config hash is comparable over time.
_STAGED_MODEL = "model.bin"
_STAGED_SHORTLIST = "lex.bin"
_STAGED_SRC_VOCAB = "vocab.src.spm"
_STAGED_TRG_VOCAB = "vocab.trg.spm"

_CONFIG_NAME = "bergamot.config.yml"
_INPUT_NAME = "input.txt"
_OUTPUT_NAME = "output.txt"

_LOG_TAIL = 4000


def _render_config(decode: DecodeConfig, gemm_precision: str) -> str:
    """
    Renders the bergamot model config for the staged work directory.

    The scalar settings are the fixed production decode settings (see
    config.py); paths are the canonical staged names, resolved by bergamot
    relative to the work directory it is invoked in.
    """
    lines = [
        f"bergamot-mode: {decode.bergamot_mode}",
        "models:",
        f"  - model/{_STAGED_MODEL}",
        "vocabs:",
        f"  - model/{_STAGED_SRC_VOCAB}",
        f"  - model/{_STAGED_TRG_VOCAB}",
        "shortlist:",
        f"  - model/{_STAGED_SHORTLIST}",
        "  - false",
        f"beam-size: {decode.beam_size}",
        f"normalize: {decode.normalize}",
        f"word-penalty: {decode.word_penalty}",
        f"max-length-break: {decode.max_length_break}",
        f"mini-batch-words: {decode.mini_batch_words}",
        f"workspace: {decode.workspace}",
        f"max-length-factor: {decode.max_length_factor}",
        f"skip-cost: {'true' if decode.skip_cost else 'false'}",
        f"cpu-threads: {decode.cpu_threads}",
        "quiet: false",
        "quiet-translation: false",
        f"gemm-precision: {gemm_precision}",
        f"alignment: {decode.alignment}",
        f"ssplit-mode: {decode.ssplit_mode}",
        f"max-input-sentence-tokens: {decode.max_input_sentence_tokens}",
        f"max-input-tokens: {decode.max_input_tokens}",
    ]
    return "\n".join(lines) + "\n"


def _decode_record(decode: DecodeConfig, gemm_precision: str) -> Dict[str, object]:
    """The decode settings recorded into the result JSON, in config spelling."""
    return {
        "bergamot-mode": decode.bergamot_mode,
        "beam-size": decode.beam_size,
        "normalize": decode.normalize,
        "word-penalty": decode.word_penalty,
        "max-length-break": decode.max_length_break,
        "mini-batch-words": decode.mini_batch_words,
        "workspace": decode.workspace,
        "max-length-factor": decode.max_length_factor,
        "skip-cost": decode.skip_cost,
        "cpu-threads": decode.cpu_threads,
        "gemm-precision": gemm_precision,
        "alignment": decode.alignment,
        "ssplit-mode": decode.ssplit_mode,
        "max-input-sentence-tokens": decode.max_input_sentence_tokens,
        "max-input-tokens": decode.max_input_tokens,
    }


def _read_lines(path: Path) -> List[str]:
    """Reads an engine output file into lines, tolerating one trailing newline."""
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[: -len("\n")]
    return text.split("\n") if text else []


class BergamotEngine:
    """
    The upstream bergamot-translator CLI as the v1 reference runtime.

    Either ``binary`` (a locally built bergamot executable) or ``image`` (a
    container image whose default command is the bergamot CLI, invoked via
    ``docker run`` with the work directory mounted at /work) is required.
    """

    name = "bergamot-translator"

    def __init__(
        self,
        binary: Optional[Path] = None,
        image: Optional[str] = None,
        decode: DecodeConfig = DEFAULT_DECODE_CONFIG,
    ):
        if (binary is None) == (image is None):
            raise EngineError(
                "configure exactly one bergamot runtime: --bergamot_bin for a "
                "local binary or --bergamot_image for a container image "
                "(env: VERSTA_BERGAMOT_BIN / VERSTA_BERGAMOT_IMAGE)"
            )
        self._binary = binary
        self._image = image
        self._decode = decode
        self._version: Optional[str] = None

    @property
    def variant(self) -> str:
        return "docker" if self._image is not None else "local"

    def _local_binary(self) -> Path:
        if self._binary is None or not self._binary.is_file():
            raise EngineError(
                f"bergamot binary not found: {self._binary} — build "
                f"bergamot-translator or pass --bergamot_image instead"
            )
        return self._binary

    def version(self) -> str:
        """
        Probes the engine's version string (marian/bergamot ``--version``).
        A failed probe never fails an eval; it records "unknown".
        """
        if self._version is not None:
            return self._version

        probed = "unknown"
        try:
            if self._image is not None:
                # The image's default command is the bergamot CLI (see the
                # class docstring), so --version reaches it directly.
                argv = ["docker", "run", "--rm", self._image, "--version"]
            else:
                argv = [str(self._local_binary()), "--version"]
            completed = subprocess.run(argv, capture_output=True, timeout=60)
            output = (completed.stdout or completed.stderr).decode(
                "utf-8", errors="replace"
            )
            first_line = output.strip().splitlines()
            if completed.returncode == 0 and first_line:
                probed = first_line[0].strip()
        except (OSError, subprocess.TimeoutExpired, EngineError):
            pass

        self._version = probed
        return probed

    def _translate_argv(self, work_dir: Path) -> List[str]:
        cli_args = [
            "--model-config-paths",
            _CONFIG_NAME,
            "--log-level",
            "warn",
        ]
        if self._image is not None:
            return [
                "docker",
                "run",
                "--rm",
                "-i",
                "-v",
                f"{work_dir}:/work",
                "-w",
                "/work",
                self._image,
                *cli_args,
            ]
        return [str(self._local_binary()), *cli_args]

    def _stage(self, direction: DirectionInput, work_dir: Path) -> str:
        """
        Stages the model payloads under canonical names inside the work
        directory and writes the fixed decode config. Returns the resolved
        gemm-precision (from the original model file name).
        """
        gemm_precision = resolve_gemm_precision(direction.model_path.name)

        model_dir = work_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(direction.model_path, model_dir / _STAGED_MODEL)
        shutil.copyfile(direction.vocabulary_path, model_dir / _STAGED_SRC_VOCAB)
        target_vocab = direction.target_vocabulary_path or direction.vocabulary_path
        shutil.copyfile(target_vocab, model_dir / _STAGED_TRG_VOCAB)
        shutil.copyfile(direction.shortlist_path, model_dir / _STAGED_SHORTLIST)

        config_text = _render_config(self._decode, gemm_precision)
        (work_dir / _CONFIG_NAME).write_text(config_text, encoding="utf-8")
        return gemm_precision

    def translate(
        self,
        direction: DirectionInput,
        sentences: List[str],
        work_dir: Path,
    ) -> TranslationOutcome:
        gemm_precision = self._stage(direction, work_dir)
        config_path = work_dir / _CONFIG_NAME

        input_path = work_dir / _INPUT_NAME
        output_path = work_dir / _OUTPUT_NAME
        input_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")

        argv = self._translate_argv(work_dir)
        try:
            with open(input_path, "rb") as stdin, open(output_path, "wb") as stdout:
                completed = subprocess.run(
                    argv,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=subprocess.PIPE,
                    cwd=None if self._image else work_dir,
                )
        except OSError as exc:
            raise EngineError(f"failed to start engine ({argv[0]}): {exc}")

        if completed.returncode != 0:
            stderr_tail = completed.stderr.decode("utf-8", errors="replace")[
                -_LOG_TAIL:
            ]
            raise EngineError(
                f"engine exited with code {completed.returncode}; "
                f"stderr tail:\n{stderr_tail}"
            )

        if not output_path.is_file():
            raise EngineError("engine produced no output file")

        translations = _read_lines(output_path)
        if len(translations) != len(sentences):
            raise EngineError(
                f"engine output is not line-aligned with the input: "
                f"{len(sentences)} sentences in, {len(translations)} lines out "
                f"(scores would be meaningless; treating as an engine fault)"
            )

        return TranslationOutcome(
            translation_path=output_path,
            translations=translations,
            version=self.version(),
            variant=self.variant,
            image=self._image,
            decode=_decode_record(self._decode, gemm_precision),
            config_sha256=sha256(config_path.read_bytes()).hexdigest(),
        )


def create_engine(
    name: str,
    bergamot_binary: Optional[Path] = None,
    bergamot_image: Optional[str] = None,
) -> Engine:
    """
    Constructs the engine selected on the command line.

    Args:
        name (str): Engine identifier; only ``bergamot`` is usable until the
            LeanMT batch CLI lands (NEU-50).
        bergamot_binary (Path | None): Local bergamot executable.
        bergamot_image (str | None): Container image for the bergamot CLI.

    Returns:
        Engine: The configured engine.

    Raises:
        EngineError: On an unknown engine name or invalid configuration.
    """
    if name == "bergamot":
        return BergamotEngine(binary=bergamot_binary, image=bergamot_image)
    if name == "leanmt":
        # Reserved for the LeanMT batch translate CLI (NEU-50), landing under
        # a versioned spec — same handshake style as NEU-25. Until then the
        # harness runs on the upstream bergamot runtime.
        raise EngineError(
            "engine 'leanmt' is reserved for the LeanMT batch translate CLI "
            "(NEU-50), which has not landed yet — use --engine bergamot"
        )
    raise EngineError(
        f"unknown engine '{name}'; available: bergamot, leanmt (reserved)"
    )
