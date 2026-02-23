#!/usr/bin/env bash
# =============================================================================
# WavBench Full Pipeline — Gemini 3.1 Pro
#
# Runs: Setup → Inference → Evaluation → Statistics
#
# Usage:
#   export GOOGLE_API_KEY="your-api-key"
#   export WAVBENCH_DIR="D:/Zhenhe/Coding/Projects/WavBench/WavBench_download"
#   bash run_pipeline.sh
#
#   # Run only a specific dataset:
#   DATA=basic_creative bash run_pipeline.sh
#
#   # Dry run (validate paths, no API calls):
#   DRY_RUN=1 bash run_pipeline.sh
#
# Environment variables:
#   GOOGLE_API_KEY       (required) Gemini API key
#   WAVBENCH_DIR         (required) Path to WavBench_download root folder
#                          Windows: D:/Zhenhe/Coding/Projects/WavBench/WavBench_download
#                          Linux:   /data/WavBench_download
#   DATA                 Dataset name or 'all'        (default: all)
#   OUTPUT_DIR           Predictions output dir        (default: ./predictions)
#   EVAL_DIR             Evaluation results dir        (default: ./eval_results)
#   STATS_FILE           Statistics output file        (default: ./statistics.txt)
#   MODEL_ID             Gemini model ID               (default: [次]gemini-3.1-pro-preview)
#   MAX_WORKERS          Parallel API workers          (default: 4)
#   SELF_CONSISTENCY_K   Votes for classification      (default: 3)
#   DRY_RUN              Set to 1 to skip API calls    (default: 0)
# =============================================================================
set -euo pipefail

# ── Force UTF-8 (prevents GBK decode errors on Chinese-locale servers) ────────
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-[次]gemini-3.1-pro-preview}"
DATA="${DATA:-all}"
WAVBENCH_DIR="${WAVBENCH_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./predictions}"
EVAL_DIR="${EVAL_DIR:-./eval_results}"
STATS_FILE="${STATS_FILE:-./statistics.txt}"
MAX_WORKERS="${MAX_WORKERS:-4}"
SELF_CONSISTENCY_K="${SELF_CONSISTENCY_K:-3}"
DRY_RUN="${DRY_RUN:-0}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ -z "${GOOGLE_API_KEY:-}" ]] && [[ "$DRY_RUN" != "1" ]]; then
    echo "❌ ERROR: GOOGLE_API_KEY is not set."
    echo "   Run: export GOOGLE_API_KEY='your-api-key'"
    exit 1
fi

if [[ -z "$WAVBENCH_DIR" ]]; then
    echo "❌ ERROR: WAVBENCH_DIR is not set."
    echo "   Run: export WAVBENCH_DIR='D:/Zhenhe/Coding/Projects/WavBench/WavBench_download'"
    exit 1
fi

if [[ ! -d "$WAVBENCH_DIR" ]]; then
    echo "❌ ERROR: WAVBENCH_DIR directory does not exist: $WAVBENCH_DIR"
    exit 1
fi

echo "============================================================"
echo " WavBench Pipeline — Gemini 3.1 Pro"
echo "============================================================"
echo " Model:          $MODEL_ID"
echo " Dataset:        $DATA"
echo " WavBench dir:   $WAVBENCH_DIR"
echo " Output dir:     $OUTPUT_DIR"
echo " Eval dir:       $EVAL_DIR"
echo " Workers:        $MAX_WORKERS"
echo " Consistency:    k=$SELF_CONSISTENCY_K"
echo " Dry run:        $DRY_RUN"
echo "============================================================"

# ── Step 1: Install dependencies ──────────────────────────────────────────────
echo ""
echo "📦 Step 1/4: Installing dependencies..."
pip install -q \
    soundfile \
    librosa \
    tqdm \
    pandas \
    pyarrow

# ── Step 2: Inference ─────────────────────────────────────────────────────────
echo ""
echo "🎙️  Step 2/4: Running inference..."

DRY_FLAG=""
if [[ "$DRY_RUN" == "1" ]]; then
    DRY_FLAG="--dry_run"
fi

python run_inference.py \
    --data "$DATA" \
    --wavbench_dir "$WAVBENCH_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_id "$MODEL_ID" \
    --max_workers "$MAX_WORKERS" \
    --self_consistency_k "$SELF_CONSISTENCY_K" \
    $DRY_FLAG

echo "✅ Inference complete → $OUTPUT_DIR"

# ── Step 3: Evaluation ────────────────────────────────────────────────────────
echo ""
echo "🧑‍⚖️  Step 3/4: Running evaluation..."

if [[ "$DATA" == "all" ]]; then
    # Full benchmark: run both eval types
    echo "  → Colloquial evaluation (Panels A & B)..."
    python run_evaluate.py \
        --eval_type colloquial \
        --dataset all \
        --predictions_dir "$OUTPUT_DIR" \
        --eval_dir "$EVAL_DIR"

    echo "  → Acoustic evaluation (Panels C, D & E)..."
    python run_evaluate.py \
        --eval_type acoustic \
        --dataset all \
        --predictions_dir "$OUTPUT_DIR" \
        --eval_dir "$EVAL_DIR"

else
    # Single dataset: route to the correct eval type
    if [[ "$DATA" == basic_* ]] || [[ "$DATA" == pro_* ]]; then
        echo "  → Colloquial evaluation..."
        python run_evaluate.py \
            --eval_type colloquial \
            --dataset "$DATA" \
            --predictions_dir "$OUTPUT_DIR" \
            --eval_dir "$EVAL_DIR"
    else
        echo "  → Acoustic evaluation..."
        python run_evaluate.py \
            --eval_type acoustic \
            --dataset "$DATA" \
            --predictions_dir "$OUTPUT_DIR" \
            --eval_dir "$EVAL_DIR"
    fi
fi

echo "✅ Evaluation complete → $EVAL_DIR"

# ── Step 4: Statistics ────────────────────────────────────────────────────────
echo ""
echo "📊 Step 4/4: Computing statistics..."

python run_statistics.py \
    --eval_dir "$EVAL_DIR" \
    --output "$STATS_FILE" \
    --csv

echo "✅ Statistics saved → $STATS_FILE"
echo ""
echo "============================================================"
echo " Pipeline complete!"
echo "============================================================"
cat "$STATS_FILE"
