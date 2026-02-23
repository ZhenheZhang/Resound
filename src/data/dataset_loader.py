"""
WavBench dataset loader.

Confirmed local directory structure (WavBench_download):

  WavBench_download/
  │
  ├── Basic/
  │   └── basic_class/               ← JSON named by domain stem only
  │       ├── code.json
  │       ├── creative.json
  │       ├── instruction.json
  │       ├── logic.json
  │       ├── math.json
  │       ├── qa.json
  │       └── satety.json
  │
  ├── Pro/
  │   └── pro_class/                 ← same structure as basic_class
  │       ├── code.json
  │       ├── creative.json
  │       └── ... (7 files)
  │
  ├── Colloquial_audio/              ← WAV files for Basic + Pro tasks
  │   ├── alpaca/
  │   ├── alignbench/
  │   ├── meeseeks/
  │   ├── wildspeech/
  │   └── ifeval/
  │       └── *.wav  (flat per class)
  │
  └── Acoustic/
      ├── explicit_generation/       ← WAV files for generation tasks
      ├── explicit_understanding/    ← WAV files for understanding tasks
      ├── implicit/                  ← WAV files for implicit tasks
      ├── multi/                     ← WAV files for multi-round tasks
      │   └── multi_round_understanding/
      │       └── multi_round_0000/
      │           ├── user1.wav ... user4.wav
      │           └── model1.wav ... model4.wav
      └── json/                      ← ALL acoustic JSONs, flat, full filenames
          ├── explicit_generation_accent.json  ...
          ├── explicit_understanding_emotion.json  ...
          ├── implicit_age_generation.json  ...
          ├── multi_round_generation.json
          └── multi_round_understanding.json

JSON format (confirmed from creative.json):
  [
    {
      "id": "0001",
      "spoken_instruction": "...",
      "spoken_reference":   "...",
      "audio_path": "./audio/alpaca/alpaca_0009.wav",
      "class": "alpaca"
    },
    ...
  ]

Audio path layout:
  The JSON always stores paths as  ./audio/{class}/{filename}.wav
  "audio/" does NOT sit next to the JSON file on disk.
  Each dataset entry carries an "audio_root" key that remaps
  "./audio/" to the correct real directory under WavBench_download:

    Basic / Pro tasks  →  Colloquial_audio/
    Explicit tasks     →  Acoustic/explicit_{generation|understanding}/
    Implicit tasks     →  Acoustic/implicit/
    Multi-round        →  Acoustic/multi/
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class WavBenchSample:
    """A single WavBench evaluation sample."""
    sample_id: str                  # e.g. "basic_creative_0001"
    panel: str                      # "colloquial_basic" | "colloquial_pro" |
                                    # "explicit_understanding" | "explicit_generation" |
                                    # "implicit"
    domain: Optional[str]           # colloquial domain: code/math/logic/qa/...
    attribute: Optional[str]        # acoustic attribute: emotion/gender/pitch/...
    audio_path: str                 # absolute path to the WAV file
    question: Optional[str]         # spoken_instruction text
    reference_answer: Optional[str] # spoken_reference text
    target_value: Optional[str]     # for generation tasks
    history: list = field(default_factory=list)   # for multi-turn implicit
    metadata: dict = field(default_factory=dict)  # task routing + raw fields


# ── Dataset routing table ─────────────────────────────────────────────────────
#
#  json_path   — path to the JSON metadata file, relative to wavbench_dir root
#  audio_root  — directory that contains the {class}/*.wav files,
#                relative to wavbench_dir root.
#                The JSON stores  "./audio/{class}/file.wav";
#                audio_root replaces the "./audio/" prefix so the real
#                path becomes  {wavbench_dir}/{audio_root}/{class}/file.wav
#
DATASET_NAME_MAP = {

    # ── Panel B: Colloquial Basic  ────────────────────────────────────────────
    # JSON:   Basic/basic_class/{domain}.json
    # Audio:  Colloquial_audio/{class}/*.wav
    "basic_code": {
        "panel": "colloquial_basic", "domain": "code", "task_type": "colloquial",
        "json_path":   "Basic/basic_class/code.json",
        "audio_root":  "Colloquial_audio",
    },
    "basic_creative": {
        "panel": "colloquial_basic", "domain": "creativity", "task_type": "colloquial",
        "json_path":   "Basic/basic_class/creative.json",
        "audio_root":  "Colloquial_audio",
    },
    "basic_instruction": {
        "panel": "colloquial_basic", "domain": "instruction", "task_type": "colloquial",
        "json_path":   "Basic/basic_class/instruction.json",
        "audio_root":  "Colloquial_audio",
    },
    "basic_logic": {
        "panel": "colloquial_basic", "domain": "logic", "task_type": "colloquial",
        "json_path":   "Basic/basic_class/logic.json",
        "audio_root":  "Colloquial_audio",
    },
    "basic_math": {
        "panel": "colloquial_basic", "domain": "math", "task_type": "colloquial",
        "json_path":   "Basic/basic_class/math.json",
        "audio_root":  "Colloquial_audio",
    },
    "basic_qa": {
        "panel": "colloquial_basic", "domain": "qa", "task_type": "colloquial",
        "json_path":   "Basic/basic_class/qa.json",
        "audio_root":  "Colloquial_audio",
    },
    "basic_satety": {
        "panel": "colloquial_basic", "domain": "safety", "task_type": "colloquial",
        "json_path":   "Basic/basic_class/satety.json",
        "audio_root":  "Colloquial_audio",
    },

    # ── Panel A: Colloquial Pro  ──────────────────────────────────────────────
    # JSON:   Pro/pro_class/{domain}.json
    # Audio:  Colloquial_audio/{class}/*.wav
    "pro_code": {
        "panel": "colloquial_pro", "domain": "code", "task_type": "colloquial",
        "json_path":   "Pro/pro_class/code.json",
        "audio_root":  "Colloquial_audio",
    },
    "pro_creative": {
        "panel": "colloquial_pro", "domain": "creativity", "task_type": "colloquial",
        "json_path":   "Pro/pro_class/creative.json",
        "audio_root":  "Colloquial_audio",
    },
    "pro_instruction": {
        "panel": "colloquial_pro", "domain": "instruction", "task_type": "colloquial",
        "json_path":   "Pro/pro_class/instruction.json",
        "audio_root":  "Colloquial_audio",
    },
    "pro_logic": {
        "panel": "colloquial_pro", "domain": "logic", "task_type": "colloquial",
        "json_path":   "Pro/pro_class/logic.json",
        "audio_root":  "Colloquial_audio",
    },
    "pro_math": {
        "panel": "colloquial_pro", "domain": "math", "task_type": "colloquial",
        "json_path":   "Pro/pro_class/math.json",
        "audio_root":  "Colloquial_audio",
    },
    "pro_qa": {
        "panel": "colloquial_pro", "domain": "qa", "task_type": "colloquial",
        "json_path":   "Pro/pro_class/qa.json",
        "audio_root":  "Colloquial_audio",
    },
    "pro_satety": {
        "panel": "colloquial_pro", "domain": "safety", "task_type": "colloquial",
        "json_path":   "Pro/pro_class/satety.json",
        "audio_root":  "Colloquial_audio",
    },

    # ── Panel C: Acoustic Explicit Understanding  ─────────────────────────────
    # JSON:   Acoustic/json/explicit_understanding_{attr}.json
    # Audio:  Acoustic/explicit_understanding/{class}/*.wav
    "acoustic_explicit_understanding_accent": {
        "panel": "explicit_understanding", "attribute": "accent",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_accent.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_age": {
        "panel": "explicit_understanding", "attribute": "age",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_age.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_audio": {
        "panel": "explicit_understanding", "attribute": "audio",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_audio.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_emotion": {
        "panel": "explicit_understanding", "attribute": "emotion",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_emotion.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_gender": {
        "panel": "explicit_understanding", "attribute": "gender",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_gender.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_lang": {
        "panel": "explicit_understanding", "attribute": "language",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_lang.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_music": {
        "panel": "explicit_understanding", "attribute": "music",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_music.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_pitch": {
        "panel": "explicit_understanding", "attribute": "pitch",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_pitch.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_speed": {
        "panel": "explicit_understanding", "attribute": "speed",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_speed.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },
    "acoustic_explicit_understanding_volume": {
        "panel": "explicit_understanding", "attribute": "volume",
        "task_type": "explicit_understanding",
        "json_path":   "Acoustic/json/explicit_understanding_volume.json",
        "audio_root":  "Acoustic/explicit_understanding",
    },

    # ── Panel D: Acoustic Explicit Generation  ────────────────────────────────
    # JSON:   Acoustic/json/explicit_generation_{attr}.json
    # Audio:  Acoustic/explicit_generation/{class}/*.wav
    "acoustic_explicit_generation_accent": {
        "panel": "explicit_generation", "attribute": "accent",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_accent.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_age": {
        "panel": "explicit_generation", "attribute": "age",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_age.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_audio": {
        "panel": "explicit_generation", "attribute": "audio",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_audio.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_emotion": {
        "panel": "explicit_generation", "attribute": "emotion",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_emotion.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_gender": {
        "panel": "explicit_generation", "attribute": "gender",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_gender.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_lang": {
        "panel": "explicit_generation", "attribute": "language",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_lang.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_music": {
        "panel": "explicit_generation", "attribute": "music",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_music.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_pitch": {
        "panel": "explicit_generation", "attribute": "pitch",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_pitch.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_speed": {
        "panel": "explicit_generation", "attribute": "speed",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_speed.json",
        "audio_root":  "Acoustic/explicit_generation",
    },
    "acoustic_explicit_generation_volume": {
        "panel": "explicit_generation", "attribute": "volume",
        "task_type": "explicit_generation",
        "json_path":   "Acoustic/json/explicit_generation_volume.json",
        "audio_root":  "Acoustic/explicit_generation",
    },

    # ── Panel E: Acoustic Implicit  ───────────────────────────────────────────
    # JSON:   Acoustic/json/implicit_{attr}_generation.json
    # Audio:  Acoustic/implicit/{class}/*.wav
    "acoustic_implicit_age_generation": {
        "panel": "implicit", "attribute": "age",
        "task_type": "implicit", "implicit_type": "generation",
        "json_path":   "Acoustic/json/implicit_age_generation.json",
        "audio_root":  "Acoustic/implicit",
    },
    "acoustic_implicit_emotion_generation": {
        "panel": "implicit", "attribute": "emotion",
        "task_type": "implicit", "implicit_type": "generation",
        "json_path":   "Acoustic/json/implicit_emotion_generation.json",
        "audio_root":  "Acoustic/implicit",
    },
    "acoustic_implicit_pitch_generation": {
        "panel": "implicit", "attribute": "pitch",
        "task_type": "implicit", "implicit_type": "generation",
        "json_path":   "Acoustic/json/implicit_pitch_generation.json",
        "audio_root":  "Acoustic/implicit",
    },
    "acoustic_implicit_speed_generation": {
        "panel": "implicit", "attribute": "speed",
        "task_type": "implicit", "implicit_type": "generation",
        "json_path":   "Acoustic/json/implicit_speed_generation.json",
        "audio_root":  "Acoustic/implicit",
    },
    "acoustic_implicit_understanding": {
        "panel": "implicit", "attribute": None,
        "task_type": "implicit", "implicit_type": "understanding",
        "json_path":   "Acoustic/json/implicit_understanding.json",
        "audio_root":  "Acoustic/implicit",
    },

    # ── Panel E: Multi-round  ─────────────────────────────────────────────────
    # JSON:   Acoustic/json/multi_round_{type}.json
    # Audio:  Acoustic/multi/multi_round_{type}/multi_round_NNNN/
    #           user1..4.wav  +  model1..4.wav
    "acoustic_multi_round_generation": {
        "panel": "implicit", "attribute": None,
        "task_type": "implicit", "implicit_type": "generation",
        "json_path":   "Acoustic/json/multi_round_generation.json",
        "audio_root":  "Acoustic/multi",
    },
    "acoustic_multi_round_understanding": {
        "panel": "implicit", "attribute": None,
        "task_type": "implicit", "implicit_type": "understanding",
        "json_path":   "Acoustic/json/multi_round_understanding.json",
        "audio_root":  "Acoustic/multi",
    },
}

ALL_DATASETS = list(DATASET_NAME_MAP.keys())


# ── Audio path resolution ─────────────────────────────────────────────────────

def _resolve_audio_path(raw: str, wavbench_root: Path, audio_root: str) -> str:
    """
    Resolve an audio_path field from a JSON row to an absolute path.

    The JSON stores paths as:
        ./audio/{class}/{filename}.wav

    The "./audio/" prefix does NOT correspond to a real "audio/" folder next
    to the JSON file. Instead, each dataset has an "audio_root" in
    DATASET_NAME_MAP that points to the actual WAV directory, e.g.:
        Colloquial_audio/          for Basic + Pro datasets
        Acoustic/explicit_understanding/  for understanding datasets

    Mapping:
        ./audio/{class}/{filename}.wav
            → {wavbench_root}/{audio_root}/{class}/{filename}.wav

    If the path is already absolute, it is returned unchanged.
    If no "./audio/" prefix is found, falls back to resolving relative
    to wavbench_root as-is.
    """
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)

    # Strip the leading "./audio/" or "audio/" prefix that the JSON uses
    parts = p.parts  # e.g. ('.', 'audio', 'alpaca', 'alpaca_0009.wav')
    if len(parts) >= 2 and parts[1] == "audio":
        # Drop the first two parts ("." and "audio"), keep the rest
        remainder = Path(*parts[2:])  # e.g. 'alpaca/alpaca_0009.wav'
    elif len(parts) >= 1 and parts[0] == "audio":
        remainder = Path(*parts[1:])
    else:
        # No "audio/" prefix — resolve directly under wavbench_root
        remainder = p

    return str((wavbench_root / audio_root / remainder).resolve())


# ── Row → WavBenchSample ──────────────────────────────────────────────────────

def _row_to_sample(
    row: dict,
    wavbench_root: Path,
    audio_root: str,
    dataset_name: str,
    meta: dict,
) -> WavBenchSample:
    """Convert one raw JSON dict into a WavBenchSample."""
    raw_audio = row.get("audio_path", "")
    return WavBenchSample(
        sample_id=f"{dataset_name}_{row.get('id', 'unknown')}",
        panel=meta["panel"],
        domain=meta.get("domain"),
        attribute=meta.get("attribute"),
        audio_path=_resolve_audio_path(raw_audio, wavbench_root, audio_root),
        question=(row.get("spoken_instruction")
                  or row.get("instruction")
                  or row.get("question") or ""),
        reference_answer=(row.get("spoken_reference")
                          or row.get("reference")
                          or row.get("answer") or ""),
        target_value=row.get("target_value") or row.get("target") or None,
        history=row.get("history") or [],
        metadata={
            "task_type":      meta["task_type"],
            "implicit_type":  meta.get("implicit_type"),
            "dataset_name":   dataset_name,
            "sample_id":      row.get("id"),
            "class":          row.get("class"),
            "raw_audio_path": raw_audio,
        },
    )


# ── Internal JSON array loader ────────────────────────────────────────────────

def _load_json_array(
    json_file: Path,
    wavbench_root: Path,
    audio_root: str,
    dataset_name: str,
    meta: dict,
) -> Iterator[WavBenchSample]:
    """Parse a JSON array file and yield WavBenchSample objects."""
    with open(json_file, encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {json_file}, got {type(data).__name__}."
        )

    try:
        rel = json_file.relative_to(wavbench_root)
    except ValueError:
        rel = json_file.name
    logger.info(f"  {len(data)} samples  ←  {rel}  (audio: {audio_root}/)")

    for row in data:
        if not isinstance(row, dict):
            logger.warning(f"Skipping non-dict entry: {row!r}")
            continue
        yield _row_to_sample(row, wavbench_root, audio_root, dataset_name, meta)


# ── Primary public API ────────────────────────────────────────────────────────

def load_from_local(
    dataset_name: str,
    wavbench_dir: str,
) -> Iterator[WavBenchSample]:
    """
    Load a WavBench dataset from the local WavBench_download directory.

    Args:
        dataset_name : Logical dataset name, e.g. "basic_creative",
                       "acoustic_explicit_understanding_emotion".
                       Must be a key in DATASET_NAME_MAP.
        wavbench_dir : Path to the WavBench_download root folder.
                       Windows: "D:/Zhenhe/Coding/Projects/WavBench/WavBench_download"
                       Linux:   "/data/WavBench_download"

    Yields:
        WavBenchSample objects with fully resolved absolute audio_path.

    Raises:
        ValueError        if dataset_name is not in DATASET_NAME_MAP.
        FileNotFoundError if wavbench_dir or the JSON file does not exist.
    """
    meta = DATASET_NAME_MAP.get(dataset_name)
    if meta is None:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'.\n"
            f"Valid options:\n  " + "\n  ".join(ALL_DATASETS)
        )

    wavbench_root = Path(wavbench_dir).resolve()
    if not wavbench_root.exists():
        raise FileNotFoundError(f"WavBench root not found: {wavbench_root}")

    json_file = wavbench_root / meta["json_path"]
    if not json_file.exists():
        raise FileNotFoundError(
            f"JSON file not found: {json_file}\n"
            f"Expected at: {meta['json_path']} inside {wavbench_root}"
        )

    audio_root = meta["audio_root"]
    logger.info(f"Loading '{dataset_name}'  [{meta['panel']}]")
    yield from _load_json_array(json_file, wavbench_root, audio_root, dataset_name, meta)


def load_from_json_file(
    json_file: str,
    dataset_name: str,
    wavbench_dir: Optional[str] = None,
) -> Iterator[WavBenchSample]:
    """
    Load directly from a JSON file path (no wavbench_dir layout required).

    Useful for quick ad-hoc testing:
        samples = list(load_from_json_file(
            "D:/WavBench_download/Basic/basic_class/creative.json",
            "basic_creative",
            wavbench_dir="D:/WavBench_download"
        ))
    """
    meta = DATASET_NAME_MAP.get(dataset_name)
    if meta is None:
        logger.warning(
            f"Unknown dataset name '{dataset_name}' — defaulting to "
            f"colloquial_basic/creativity."
        )
        meta = {
            "panel": "colloquial_basic", "domain": "creativity",
            "task_type": "colloquial",
            "json_path": "", "audio_root": "Colloquial_audio",
        }

    json_path = Path(json_file).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    wavbench_root = Path(wavbench_dir).resolve() if wavbench_dir else json_path.parent
    audio_root = meta.get("audio_root", "Colloquial_audio")
    yield from _load_json_array(json_path, wavbench_root, audio_root, dataset_name, meta)


# ── Parquet fallback ──────────────────────────────────────────────────────────

def load_from_parquet(
    dataset_name: str,
    wavbench_dir: str,
) -> Iterator[WavBenchSample]:
    """
    Fallback for HuggingFace-converted parquet files.
    Only needed if downloaded via `datasets` library rather than Git LFS.
    """
    import pandas as pd

    meta = DATASET_NAME_MAP.get(dataset_name)
    if meta is None:
        raise ValueError(f"Unknown dataset: '{dataset_name}'")

    wavbench_root = Path(wavbench_dir).resolve()
    json_file     = wavbench_root / meta["json_path"]
    parquet_dir   = json_file.parent
    parquet_files = sorted(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {parquet_dir}")

    audio_root = meta["audio_root"]
    logger.info(f"Loading '{dataset_name}' from {len(parquet_files)} parquet file(s)")

    for pf in parquet_files:
        df = pd.read_parquet(pf)

        # Decode byte-string columns safely (GBK-safe)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
            )

        for _, row in df.iterrows():
            row = row.to_dict()
            if "spoken_instruction" not in row:
                row["spoken_instruction"] = (
                    row.get("instruction") or row.get("question") or row.get("prompt") or ""
                )
            if "spoken_reference" not in row:
                row["spoken_reference"] = (
                    row.get("reference") or row.get("answer") or row.get("label") or ""
                )
            if "id" not in row:
                row["id"] = str(row.get("index", "unknown"))
            if "audio_path" not in row:
                row["audio_path"] = (
                    row.get("file_name") or row.get("file") or row.get("audio_file") or ""
                )
            yield _row_to_sample(row, wavbench_root, audio_root, dataset_name, meta)
