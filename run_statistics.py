#!/usr/bin/env python3
"""
WavBench statistics aggregator.

Reads per-sample *_eval.jsonl files produced by run_evaluate.py and outputs:
  1. A Table-2-format text report comparing Gemini 3.1 Pro against all five
     official leaderboard models (Qwen3-Omni, Kimi-Audio, Mimo-Audio,
     Step-Audio-2, GPT-4o Audio).
  2. A Figure-1-style radar chart — one spoke per panel, polygon per model.

Panel layout (mirrors the WavBench paper exactly):
  Panel A  Colloquial Expression - Pro subset    (score scale 0-100)
  Panel B  Colloquial Expression - Basic subset  (score scale 0-100)
  Panel C  Explicit Acoustic Understanding        (score scale 0-100)
  Panel D  Explicit Acoustic Generation           (score scale 0-100)
  Panel E  Implicit Acoustic Interaction          (score scale 0-5)

Panel E sub-task mapping (corrected):
  Single-Turn (Text)   -> acoustic_implicit_understanding          [1 dataset]
  Single-Turn (Audio)  -> grand mean of 4 implicit-gen datasets    [age/emotion/pitch/speed]
  Multi-Turn  (Text)   -> acoustic_multi_round_understanding       [1 dataset]
  Multi-Turn  (Audio)  -> acoustic_multi_round_generation          [1 dataset]

Averaging:
  Official models -- panel averages taken directly from Table 2 (sample-weighted
  grand means; domain sample counts differ so simple mean of 7 domains != official).
  Our model       -- panel averages computed as the grand mean over every individual
  eval_score value across all datasets in the panel, matching the official
  sample-weighted methodology when sample counts are known.

Usage:
    python run_statistics.py
    python run_statistics.py --eval_dir ./eval_results \\
                             --output   ./statistics/statistics.txt \\
                             --chart    ./statistics/wavbench_radar.png
    python run_statistics.py --no_chart   # text report only

Outputs are saved under ./statistics/ by default:
    ./statistics/statistics.txt    -- Table-2 text report
    ./statistics/wavbench_radar.png -- Figure-1 radar chart
"""

import json
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("statistics")


# -----------------------------------------------------------------------------
# Official leaderboard data  (Table 2, WavBench paper)
#
# Per-domain keys : direct scores from Table 2 rows
# _avg_X keys     : taken VERBATIM from Table 2 "Avg" rows --
#                   they are sample-weighted and differ from the simple mean
#                   of domain averages (sample counts per domain are unequal)
# -----------------------------------------------------------------------------

LEADERBOARD = {
    "Qwen3-Omni": {
        "pro_code": 39.75, "pro_creative": 48.39, "pro_instruction": 43.01,
        "pro_logic": 33.21, "pro_math": 38.55, "pro_qa": 50.93, "pro_satety": 60.00,
        "_avg_A": 39.53,
        "basic_code": 53.10, "basic_creative": 57.44, "basic_instruction": 57.29,
        "basic_logic": 52.35, "basic_math": 51.05, "basic_qa": 57.54, "basic_satety": 59.67,
        "_avg_B": 55.80,
        "acoustic_explicit_understanding_accent":  37.50,
        "acoustic_explicit_understanding_age":     64.33,
        "acoustic_explicit_understanding_emotion": 92.86,
        "acoustic_explicit_understanding_gender":  21.00,
        "acoustic_explicit_understanding_lang":    83.50,
        "acoustic_explicit_understanding_pitch":   32.44,
        "acoustic_explicit_understanding_speed":   46.67,
        "acoustic_explicit_understanding_volume":  33.78,
        "acoustic_explicit_understanding_audio":   61.73,
        "acoustic_explicit_understanding_music":   22.22,
        "_avg_C": 49.60,
        "acoustic_explicit_generation_accent":  37.50,
        "acoustic_explicit_generation_age":     64.65,
        "acoustic_explicit_generation_emotion": 90.04,
        "acoustic_explicit_generation_gender":  72.27,
        "acoustic_explicit_generation_lang":    89.84,
        "acoustic_explicit_generation_pitch":   76.56,
        "acoustic_explicit_generation_speed":   43.75,
        "acoustic_explicit_generation_volume":  56.25,
        "acoustic_explicit_generation_audio":   27.03,
        "acoustic_explicit_generation_music":   62.50,
        "_avg_D": 62.03,
        "acoustic_implicit_understanding":        1.85,
        "acoustic_implicit_singleturn_audio_avg": 3.17,
        "acoustic_multi_round_understanding":     4.88,
        "acoustic_multi_round_generation":        1.25,
        "_avg_E": 2.78,
    },
    "Kimi-Audio": {
        "pro_code": 30.29, "pro_creative": 31.78, "pro_instruction": 29.86,
        "pro_logic": 26.03, "pro_math": 27.30, "pro_qa": 42.54, "pro_satety": 56.19,
        "_avg_A": 30.79,
        "basic_code": 40.69, "basic_creative": 41.57, "basic_instruction": 44.41,
        "basic_logic": 50.74, "basic_math": 41.27, "basic_qa": 49.07, "basic_satety": 58.83,
        "_avg_B": 49.23,
        "acoustic_explicit_understanding_accent":  11.00,
        "acoustic_explicit_understanding_age":     53.67,
        "acoustic_explicit_understanding_emotion": 77.33,
        "acoustic_explicit_understanding_gender":  44.50,
        "acoustic_explicit_understanding_lang":    91.00,
        "acoustic_explicit_understanding_pitch":   23.11,
        "acoustic_explicit_understanding_speed":   54.67,
        "acoustic_explicit_understanding_volume":  38.22,
        "acoustic_explicit_understanding_audio":   67.90,
        "acoustic_explicit_understanding_music":   66.67,
        "_avg_C": 52.80,
        "acoustic_explicit_generation_accent":  3.52,
        "acoustic_explicit_generation_age":     46.88,
        "acoustic_explicit_generation_emotion": 50.29,
        "acoustic_explicit_generation_gender":  45.31,
        "acoustic_explicit_generation_lang":    74.80,
        "acoustic_explicit_generation_pitch":   47.27,
        "acoustic_explicit_generation_speed":   47.27,
        "acoustic_explicit_generation_volume":  64.06,
        "acoustic_explicit_generation_audio":   10.81,
        "acoustic_explicit_generation_music":   20.83,
        "_avg_D": 41.10,
        "acoustic_implicit_understanding":        1.84,
        "acoustic_implicit_singleturn_audio_avg": 3.21,
        "acoustic_multi_round_understanding":     4.57,
        "acoustic_multi_round_generation":        1.08,
        "_avg_E": 2.67,
    },
    "Mimo-Audio": {
        "pro_code": 28.96, "pro_creative": 42.86, "pro_instruction": 36.44,
        "pro_logic": 27.57, "pro_math": 25.68, "pro_qa": 41.28, "pro_satety": 56.19,
        "_avg_A": 32.02,
        "basic_code": 42.07, "basic_creative": 45.29, "basic_instruction": 33.56,
        "basic_logic": 49.91, "basic_math": 38.73, "basic_qa": 49.12, "basic_satety": 62.83,
        "_avg_B": 49.57,
        "acoustic_explicit_understanding_accent":  27.00,
        "acoustic_explicit_understanding_age":     53.00,
        "acoustic_explicit_understanding_emotion": 77.33,
        "acoustic_explicit_understanding_gender":  20.00,
        "acoustic_explicit_understanding_lang":    53.50,
        "acoustic_explicit_understanding_pitch":   24.00,
        "acoustic_explicit_understanding_speed":   48.89,
        "acoustic_explicit_understanding_volume":  31.11,
        "acoustic_explicit_understanding_audio":   19.75,
        "acoustic_explicit_understanding_music":   55.56,
        "_avg_C": 41.02,
        "acoustic_explicit_generation_accent":  23.44,
        "acoustic_explicit_generation_age":     51.95,
        "acoustic_explicit_generation_emotion": 57.13,
        "acoustic_explicit_generation_gender":  67.58,
        "acoustic_explicit_generation_lang":    51.56,
        "acoustic_explicit_generation_pitch":   80.27,
        "acoustic_explicit_generation_speed":   51.56,
        "acoustic_explicit_generation_volume":  59.96,
        "acoustic_explicit_generation_audio":    9.46,
        "acoustic_explicit_generation_music":   16.67,
        "_avg_D": 46.93,
        "acoustic_implicit_understanding":        2.23,
        "acoustic_implicit_singleturn_audio_avg": 2.47,
        "acoustic_multi_round_understanding":     4.61,
        "acoustic_multi_round_generation":        1.04,
        "_avg_E": 2.59,
    },
    "Step-Audio-2": {
        "pro_code": 31.20, "pro_creative": 35.00, "pro_instruction": 29.40,
        "pro_logic": 26.20, "pro_math": 22.40, "pro_qa": 40.80, "pro_satety": 52.40,
        "_avg_A": 30.40,
        "basic_code": 37.20, "basic_creative": 47.20, "basic_instruction": 36.60,
        "basic_logic": 48.80, "basic_math": 30.20, "basic_qa": 48.60, "basic_satety": 60.20,
        "_avg_B": 48.50,
        "acoustic_explicit_understanding_accent":  20.67,
        "acoustic_explicit_understanding_age":     67.67,
        "acoustic_explicit_understanding_emotion": 75.43,
        "acoustic_explicit_understanding_gender":  68.00,
        "acoustic_explicit_understanding_lang":    96.50,
        "acoustic_explicit_understanding_pitch":   34.22,
        "acoustic_explicit_understanding_speed":   44.00,
        "acoustic_explicit_understanding_volume":  50.67,
        "acoustic_explicit_understanding_audio":   39.51,
        "acoustic_explicit_understanding_music":   77.78,
        "_avg_C": 57.36,
        "acoustic_explicit_generation_accent":  22.07,
        "acoustic_explicit_generation_age":     31.64,
        "acoustic_explicit_generation_emotion": 66.50,
        "acoustic_explicit_generation_gender":  59.77,
        "acoustic_explicit_generation_lang":    91.41,
        "acoustic_explicit_generation_pitch":   55.66,
        "acoustic_explicit_generation_speed":   69.14,
        "acoustic_explicit_generation_volume":  57.03,
        "acoustic_explicit_generation_audio":   32.43,
        "acoustic_explicit_generation_music":   70.83,
        "_avg_D": 55.65,
        "acoustic_implicit_understanding":        1.12,
        "acoustic_implicit_singleturn_audio_avg": 3.50,
        "acoustic_multi_round_understanding":     4.38,
        "acoustic_multi_round_generation":        1.21,
        "_avg_E": 2.55,
    },
    "GPT-4o Audio": {
        "pro_code": 53.60, "pro_creative": 63.00, "pro_instruction": 57.80,
        "pro_logic": 42.60, "pro_math": 50.20, "pro_qa": 72.80, "pro_satety": 67.60,
        "_avg_A": 58.23,
        "basic_code": 58.00, "basic_creative": 71.20, "basic_instruction": 66.80,
        "basic_logic": 67.00, "basic_math": 62.40, "basic_qa": 75.60, "basic_satety": 81.00,
        "_avg_B": 68.80,
        "acoustic_explicit_understanding_accent":  15.67,
        "acoustic_explicit_understanding_age":     20.33,
        "acoustic_explicit_understanding_emotion": 85.90,
        "acoustic_explicit_understanding_gender":  61.50,
        "acoustic_explicit_understanding_lang":    97.00,
        "acoustic_explicit_understanding_pitch":   23.56,
        "acoustic_explicit_understanding_speed":   48.00,
        "acoustic_explicit_understanding_volume":  41.78,
        "acoustic_explicit_understanding_audio":   59.26,
        "acoustic_explicit_understanding_music":   33.33,
        "_avg_C": 48.70,
        "acoustic_explicit_generation_accent":  74.22,
        "acoustic_explicit_generation_age":     78.12,
        "acoustic_explicit_generation_emotion": 95.51,
        "acoustic_explicit_generation_gender":  98.83,
        "acoustic_explicit_generation_lang":    87.89,
        "acoustic_explicit_generation_pitch":   85.74,
        "acoustic_explicit_generation_speed":   66.60,
        "acoustic_explicit_generation_volume":  82.42,
        "acoustic_explicit_generation_audio":   45.95,
        "acoustic_explicit_generation_music":   77.08,
        "_avg_D": 79.23,
        "acoustic_implicit_understanding":        2.43,
        "acoustic_implicit_singleturn_audio_avg": 2.96,
        "acoustic_multi_round_understanding":     4.48,
        "acoustic_multi_round_generation":        1.23,
        "_avg_E": 2.78,
    },
}

MODEL_ORDER = ["Qwen3-Omni", "Kimi-Audio", "Mimo-Audio", "Step-Audio-2", "GPT-4o Audio"]
OUR_MODEL   = "[Resound]Gemini 3.1 Pro"


# -----------------------------------------------------------------------------
# Panel definitions
# -----------------------------------------------------------------------------

PANELS = {
    "A": {
        "title": "Panel A: Colloquial Expression (Pro)",
        "scale": 100,
        "datasets": [
            ("pro_code",        "Code"),
            ("pro_creative",    "Creativity"),
            ("pro_instruction", "Instruction"),
            ("pro_logic",       "Logic"),
            ("pro_math",        "Math"),
            ("pro_qa",          "QA"),
            ("pro_satety",      "Safety"),
        ],
    },
    "B": {
        "title": "Panel B: Colloquial Expression (Basic)",
        "scale": 100,
        "datasets": [
            ("basic_code",        "Code"),
            ("basic_creative",    "Creativity"),
            ("basic_instruction", "Instruction"),
            ("basic_logic",       "Logic"),
            ("basic_math",        "Math"),
            ("basic_qa",          "QA"),
            ("basic_satety",      "Safety"),
        ],
    },
    "C": {
        "title": "Panel C: Explicit Acoustic Understanding",
        "scale": 100,
        "datasets": [
            ("acoustic_explicit_understanding_accent",  "Accent"),
            ("acoustic_explicit_understanding_age",     "Age"),
            ("acoustic_explicit_understanding_emotion", "Emotion"),
            ("acoustic_explicit_understanding_gender",  "Gender"),
            ("acoustic_explicit_understanding_lang",    "Language"),
            ("acoustic_explicit_understanding_pitch",   "Pitch"),
            ("acoustic_explicit_understanding_speed",   "Speed"),
            ("acoustic_explicit_understanding_volume",  "Volume"),
            ("acoustic_explicit_understanding_audio",   "Audio Event"),
            ("acoustic_explicit_understanding_music",   "Music"),
        ],
    },
    "D": {
        "title": "Panel D: Explicit Acoustic Generation",
        "scale": 100,
        "datasets": [
            ("acoustic_explicit_generation_accent",  "Accent"),
            ("acoustic_explicit_generation_age",     "Age"),
            ("acoustic_explicit_generation_emotion", "Emotion"),
            ("acoustic_explicit_generation_gender",  "Gender"),
            ("acoustic_explicit_generation_lang",    "Language"),
            ("acoustic_explicit_generation_pitch",   "Pitch"),
            ("acoustic_explicit_generation_speed",   "Speed"),
            ("acoustic_explicit_generation_volume",  "Volume"),
            ("acoustic_explicit_generation_audio",   "Audio"),
            ("acoustic_explicit_generation_music",   "Music"),
        ],
    },
    "E": {
        "title": "Panel E: Implicit Acoustic Interaction",
        "scale": 5,
        "datasets": [
            ("acoustic_implicit_understanding",        "Single-Turn (Text)"),
            ("acoustic_implicit_singleturn_audio_avg", "Single-Turn (Audio)"),
            ("acoustic_multi_round_understanding",     "Multi-Turn (Text)"),
            ("acoustic_multi_round_generation",        "Multi-Turn (Audio)"),
        ],
        "singleturn_audio_datasets": [
            "acoustic_implicit_age_generation",
            "acoustic_implicit_emotion_generation",
            "acoustic_implicit_pitch_generation",
            "acoustic_implicit_speed_generation",
        ],
    },
}


# -----------------------------------------------------------------------------
# Load our eval scores
# -----------------------------------------------------------------------------

def load_our_scores(eval_dir: Path) -> dict:
    """
    Read every *_eval.jsonl and return:
      per_dataset -- {ds_name: [float ...]}  all individual eval_score values
      averages    -- {ds_name: float}         grand mean per dataset

    Also synthesises "acoustic_implicit_singleturn_audio_avg" as the grand
    mean across all 4 implicit-generation dataset samples.
    """
    per_dataset = {}

    for eval_file in sorted(eval_dir.glob("*_eval.jsonl")):
        ds_name = eval_file.stem.replace("_eval", "")
        scores = []
        with open(eval_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line).get("eval_score")
                    if s is not None:
                        scores.append(float(s))
                except Exception:
                    pass
        if scores:
            per_dataset[ds_name] = scores

    averages = {ds: sum(sc) / len(sc) for ds, sc in per_dataset.items()}

    # Synthetic Single-Turn (Audio): grand mean over 4 raw dataset samples
    st_audio_raw = []
    for ds in PANELS["E"]["singleturn_audio_datasets"]:
        st_audio_raw.extend(per_dataset.get(ds, []))
    if st_audio_raw:
        averages["acoustic_implicit_singleturn_audio_avg"] = (
            sum(st_audio_raw) / len(st_audio_raw)
        )
        n_ds = sum(1 for ds in PANELS["E"]["singleturn_audio_datasets"] if ds in per_dataset)
        logger.debug(
            f"  Single-Turn (Audio): {averages['acoustic_implicit_singleturn_audio_avg']:.4f} "
            f"({n_ds}/4 datasets, {len(st_audio_raw)} samples)"
        )

    return {"per_dataset": per_dataset, "averages": averages}


def get_panel_avg_ours(per_dataset: dict, panel_key: str):
    """
    Grand mean over ALL individual eval_score values across every dataset in
    the panel. Matches the official sample-weighted methodology.

    For Panel E, uses the raw implicit-generation datasets directly to avoid
    double-counting with the synthetic average key.
    """
    all_scores = []
    if panel_key == "E":
        raw_keys = (
            ["acoustic_implicit_understanding",
             "acoustic_multi_round_understanding",
             "acoustic_multi_round_generation"]
            + PANELS["E"]["singleturn_audio_datasets"]
        )
        for ds in raw_keys:
            all_scores.extend(per_dataset.get(ds, []))
    else:
        for ds, _ in PANELS[panel_key]["datasets"]:
            all_scores.extend(per_dataset.get(ds, []))
    return sum(all_scores) / len(all_scores) if all_scores else None


# -----------------------------------------------------------------------------
# Text report (Table 2 format)
# -----------------------------------------------------------------------------

def build_report(our_data: dict) -> str:
    averages    = our_data["averages"]
    per_dataset = our_data["per_dataset"]
    all_models  = MODEL_ORDER + [OUR_MODEL]
    col_w = 16

    def fmt(v):
        return "N/A".rjust(col_w) if v is None else f"{v:.2f}".rjust(col_w)

    W    = 32 + col_w * len(all_models)
    sep  = "-" * W
    sep2 = "." * W
    top  = "=" * W

    lines = [
        top,
        "WavBench Evaluation  --  Table 2 Format",
        f"Our model : {OUR_MODEL}",
        top,
        f"  {'Metric / Task':<30}" + "".join(m.rjust(col_w) for m in all_models),
    ]

    for panel_key, panel in PANELS.items():
        scale = panel["scale"]
        lines += [sep, f"  {panel['title']}", sep2]

        for ds_key, label in panel["datasets"]:
            row = f"  {label:<30}"
            for m in MODEL_ORDER:
                row += fmt(LEADERBOARD[m].get(ds_key))
            row += fmt(averages.get(ds_key))
            lines.append(row)

        lines.append(sep2)
        scale_tag = "  (0-5)" if scale == 5 else ""
        avg_row   = f"  {'Avg' + scale_tag:<30}"
        for m in MODEL_ORDER:
            avg_row += fmt(LEADERBOARD[m].get(f"_avg_{panel_key}"))
        avg_row += fmt(get_panel_avg_ours(per_dataset, panel_key))
        lines.append(avg_row)

    lines += [
        sep,
        "  * Panel E scores are on a 0-5 scale; all others are 0-100.",
        "  * Official model averages are sample-weighted grand means (Table 2).",
        "  * Our averages are grand means over all individual eval_score values.",
        top,
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Radar chart (Figure 1 style)
# -----------------------------------------------------------------------------

MODEL_COLORS = {
    "Qwen3-Omni":   "#4C72B0",
    "Kimi-Audio":   "#DD8452",
    "Mimo-Audio":   "#55A868",
    "Step-Audio-2": "#C44E52",
    "GPT-4o Audio": "#8172B2",
    OUR_MODEL:      "#E6AC00",
}


def build_radar_chart(our_data: dict, output_path: Path):
    """
    Pentagon radar chart - one spoke per panel (A -> B -> C -> D -> E),
    clockwise from the top.

    All axes run 0-100.  Panel E's 0-5 scores are multiplied by 20 for the
    polygon; the spoke label states the true scale.  Each vertex of OUR polygon
    is annotated with the native-scale score for easy reading.
    """
    per_dataset = our_data["per_dataset"]
    panel_keys  = ["A", "B", "C", "D", "E"]

    spoke_labels = [
        "Colloquial\n(Pro)",
        "Colloquial\n(Basic)",
        "Explicit\nUnderstanding",
        "Explicit\nGeneration",
        "Implicit\n(0-5)",
    ]

    N = len(panel_keys)
    # Clockwise from top: theta = pi/2, pi/2 - 2pi/5, pi/2 - 4pi/5, ...
    angles = [(np.pi / 2 - i * 2 * np.pi / N) % (2 * np.pi) for i in range(N)]
    angles_closed = angles + [angles[0]]

    def avg_norm(model_key, pk):
        """Panel average on 0-100 display scale."""
        if model_key == OUR_MODEL:
            v = get_panel_avg_ours(per_dataset, pk)
        else:
            v = LEADERBOARD[model_key].get(f"_avg_{pk}")
        if v is None:
            return 0.0
        return v * 20 if PANELS[pk]["scale"] == 5 else v

    def avg_display(model_key, pk):
        """Native-scale value as string for annotation."""
        if model_key == OUR_MODEL:
            v = get_panel_avg_ours(per_dataset, pk)
        else:
            v = LEADERBOARD[model_key].get(f"_avg_{pk}")
        return "N/A" if v is None else f"{v:.1f}"

    # ---- figure setup -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8F8F8")
    ax.set_theta_zero_location("N")   # 0 deg at top
    ax.set_theta_direction(-1)        # clockwise

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"],
                       fontsize=7, color="#BBBBBB")
    ax.yaxis.set_tick_params(pad=2)
    ax.grid(color="#DDDDDD", linewidth=0.7, linestyle="-", alpha=0.8)
    ax.spines["polar"].set_visible(False)

    ax.set_xticks([np.pi / 2 - i * 2 * np.pi / N for i in range(N)])
    ax.set_xticklabels(spoke_labels, fontsize=10, fontweight="bold", color="#333333")
    ax.tick_params(axis="x", pad=13)

    # ---- draw polygons ------------------------------------------------------
    all_models = MODEL_ORDER + [OUR_MODEL]
    for m in all_models:
        vals = [avg_norm(m, pk) for pk in panel_keys]
        vals_closed = vals + [vals[0]]
        color = MODEL_COLORS[m]
        is_ours = (m == OUR_MODEL)
        ax.plot(angles_closed, vals_closed,
                color=color,
                linewidth=2.8 if is_ours else 1.5,
                linestyle="--" if is_ours else "-",
                zorder=10 if is_ours else 3)
        ax.fill(angles_closed, vals_closed,
                alpha=0.20 if is_ours else 0.07,
                color=color,
                zorder=9 if is_ours else 2)

    # ---- vertex labels for our model ----------------------------------------
    for i, pk in enumerate(panel_keys):
        r    = avg_norm(OUR_MODEL, pk)
        disp = avg_display(OUR_MODEL, pk)
        if r == 0.0 and disp == "N/A":
            continue
        ang     = angles[i]
        r_label = min(r + 6, 96)
        ax.annotate(
            disp,
            xy=(ang, r), xytext=(ang, r_label),
            fontsize=8.5, fontweight="bold",
            color=MODEL_COLORS[OUR_MODEL],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=MODEL_COLORS[OUR_MODEL], lw=0.9, alpha=0.92),
            zorder=20,
        )

    # ---- legend -------------------------------------------------------------
    handles = [
        plt.Line2D([0], [0],
                   color=MODEL_COLORS[m],
                   linewidth=2.8 if m == OUR_MODEL else 1.5,
                   linestyle="--" if m == OUR_MODEL else "-",
                   label=(f"{m}  (Ours)" if m == OUR_MODEL else m))
        for m in all_models
    ]
    ax.legend(handles=handles,
              loc="lower left", bbox_to_anchor=(-0.22, -0.13),
              fontsize=9, framealpha=0.95, edgecolor="#CCCCCC",
              title="Models", title_fontsize=9)

    # ---- title & footer -----------------------------------------------------
    ax.set_title(
        "WavBench Overview Results\n"
        "Panel averages across 5 evaluation dimensions",
        fontsize=12, fontweight="bold", pad=20, color="#111111",
    )
    fig.text(
        0.5, 0.005,
        ("Panels A-D: 0-100  |  Panel E (Implicit): native scale 0-5, "
         "normalised x20 for display  |  Labels show our model scores"),
        ha="center", fontsize=7, color="#AAAAAA",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Radar chart saved -> {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate WavBench eval results and produce:\n"
            "  1. A Table-2-format text report\n"
            "  2. A Figure-1-style radar chart PNG"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval_dir", default="./eval_results",
                        help="Directory containing *_eval.jsonl files "
                             "(default: ./eval_results)")
    parser.add_argument("--output",   default="./statistics/statistics.txt",
                        help="Output path for the text report "
                             "(default: ./statistics/statistics.txt)")
    parser.add_argument("--chart",    default="./statistics/wavbench_radar.png",
                        help="Output path for the radar chart PNG "
                             "(default: ./statistics/wavbench_radar.png)")
    parser.add_argument("--no_chart", action="store_true",
                        help="Skip radar chart generation")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        logger.error(f"eval_dir not found: {eval_dir}")
        return

    our_data = load_our_scores(eval_dir)
    if not our_data["averages"]:
        logger.error("No evaluation results found.  Run run_evaluate.py first.")
        return

    n_ds = len(our_data["averages"])
    n_panels = sum(
        1 for pk in PANELS
        if any(ds in our_data["averages"]
               for ds, _ in PANELS[pk]["datasets"]
               if ds != "acoustic_implicit_singleturn_audio_avg")
    )
    logger.info(f"Loaded {n_ds} dataset averages -> {n_panels}/5 panels covered")

    # Text report
    report = build_report(our_data)
    print(report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Text report  -> {output_path}")

    # Radar chart
    if not args.no_chart:
        chart_path = Path(args.chart)
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        build_radar_chart(our_data, chart_path)


if __name__ == "__main__":
    main()
