#!/usr/bin/env bash
# Sequential rebuild of all four Murray indexes with aspect_preserve_pad_v1
# preprocessing, latitude shuffle, void-tile filter, and capped nlist.
# Run via: nohup ./scripts/run_aspectpad_rebuild.sh > outputs/rebuild_logs/driver.log 2>&1 &

set -e

cd "$(dirname "$0")/.."

OUT="outputs/murray_rebuild_aspectpad"
LOG="outputs/rebuild_logs"
mkdir -p "$OUT" "$LOG"

PY=".pixi/envs/default/bin/python"
export HF_HOME=/mnt/bigdisk/hf_cache
export HF_HUB_CACHE=/mnt/bigdisk/hf_cache/hub
export HF_TOKEN=$(cat /home/ackermand@hhmi.org/.cache/huggingface/token 2>/dev/null || echo "")

echo "=== $(date -Iseconds): START z8 CLS ==="
$PY scripts/stream_murray_index.py \
    --zoom 8 \
    --concurrency 32 \
    --batch-size 128 \
    --train-sample 20000 \
    --pq-bytes 64 \
    --output-dir "$OUT/z8_cls" \
    > "$LOG/z8_cls.log" 2>&1
echo "=== $(date -Iseconds): DONE z8 CLS ==="

echo "=== $(date -Iseconds): START z10 CLS ==="
$PY scripts/stream_murray_index.py \
    --zoom 10 \
    --concurrency 32 \
    --batch-size 128 \
    --train-sample 200000 \
    --pq-bytes 64 \
    --output-dir "$OUT/z10_cls" \
    > "$LOG/z10_cls.log" 2>&1
echo "=== $(date -Iseconds): DONE z10 CLS ==="

echo "=== $(date -Iseconds): START z8 PATCH ==="
$PY scripts/stream_murray_patch_index.py \
    --zoom 8 \
    --concurrency 48 \
    --batch-size 64 \
    --train-sample 200000 \
    --pq-bytes 64 \
    --output-dir "$OUT/z8_patch" \
    > "$LOG/z8_patch.log" 2>&1
echo "=== $(date -Iseconds): DONE z8 PATCH ==="

echo "=== $(date -Iseconds): START z10 PATCH ==="
$PY scripts/stream_murray_patch_index.py \
    --zoom 10 \
    --concurrency 48 \
    --batch-size 64 \
    --train-sample 200000 \
    --pq-bytes 64 \
    --output-dir "$OUT/z10_patch" \
    > "$LOG/z10_patch.log" 2>&1
echo "=== $(date -Iseconds): DONE z10 PATCH ==="

echo "=== $(date -Iseconds): ALL REBUILDS COMPLETE ==="

# Verify each sidecar carries the new preprocess flag.
for d in z8_cls z10_cls z8_patch z10_patch; do
    if [ -f "$OUT/$d/faiss.model.json" ]; then
        sidecar="$OUT/$d/faiss.model.json"
    else
        sidecar="$OUT/$d/patches.model.json"
    fi
    echo "--- $d sidecar:"
    grep -E '"aspect_corrected"|"preprocess"|"nlist"' "$sidecar" || echo "  (no sidecar found at $sidecar)"
done
