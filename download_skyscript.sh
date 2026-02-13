#!/usr/bin/env bash
# download_skyscript.sh
# Download SkyScript dataset (images, meta, CSV captions, benchmarks, checkpoints)
# Usage:
#   bash download_skyscript.sh
# Optional env:
#   SKYSCRIPT_DIR=/path/to/save bash download_skyscript.sh
#   DO_UNZIP=0 bash download_skyscript.sh   # disable unzip

set -euo pipefail

BASE="https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript"
ROOT="/home/hungvu/code/khanh/SkyScript/data"
DO_UNZIP="${DO_UNZIP:-1}"

mkdir -p "$ROOT"/{images,meta,dataframe,benchmark,ckpt}

download() {
  local url="$1"
  local out="$2"
  echo "==> $out"
  mkdir -p "$(dirname "$out")"
  curl -fL -C - --retry 5 --retry-delay 5 --retry-connrefused \
    -o "$out" \
    "$url"
}

maybe_unzip() {
  local zipfile="$1"
  local outdir="$2"
  if [[ "$DO_UNZIP" == "1" ]]; then
    mkdir -p "$outdir"
    echo "==> Unzip $(basename "$zipfile") -> $outdir"
    unzip -q "$zipfile" -d "$outdir"
  fi
}

echo "Saving to: $ROOT"
echo "Unzip: $DO_UNZIP"
echo

# -------- 1) Images: images2.zip ... images7.zip --------
for i in {2..7}; do
  f="images${i}.zip"
  download "$BASE/$f" "$ROOT/images/$f"
done

if [[ "$DO_UNZIP" == "1" ]]; then
  for i in {2..7}; do
    zipfile="$ROOT/images/images${i}.zip"
    outdir="$ROOT/images/images${i}"
    maybe_unzip "$zipfile" "$outdir"
  done
fi

# -------- 2) Meta: meta2.zip ... meta7.zip --------
for i in {2..7}; do
  f="meta${i}.zip"
  download "$BASE/$f" "$ROOT/meta/$f"
done

if [[ "$DO_UNZIP" == "1" ]]; then
  for i in {2..7}; do
    zipfile="$ROOT/meta/meta${i}.zip"
    outdir="$ROOT/meta/meta${i}"
    maybe_unzip "$zipfile" "$outdir"
  done
fi

# -------- 3) Captions CSVs --------
CSV_FILES=(
  "SkyScript_train_unfiltered_5M.csv"
  "SkyScript_train_top30pct_filtered_by_CLIP_openai.csv"
  "SkyScript_train_top50pct_filtered_by_CLIP_openai.csv"
  "SkyScript_val_5K_filtered_by_CLIP_openai.csv"
  "SkyScript_test_30K_filtered_by_CLIP_openai.csv"
  "SkyScript_train_top30pct_filtered_by_CLIP_laion_RS.csv"
  "SkyScript_train_top50pct_filtered_by_CLIP_laion_RS.csv"
  "SkyScript_val_5K_filtered_by_CLIP_laion_RS.csv"
  "SkyScript_test_30K_filtered_by_CLIP_laion_RS.csv"
  # language-polished (might not exist on some mirrors; we will try and skip if missing)
  "SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv"
  "SkyScript_train_top50pct_filtered_by_CLIP_laion_RS_language_polished.csv"
)

for f in "${CSV_FILES[@]}"; do
  echo "==> CSV $f"
  if ! curl -fL -C - --retry 5 --retry-delay 5 --retry-connrefused \
    -o "$ROOT/dataframe/$f" \
    "$BASE/dataframe/$f"; then
    echo "    !! Not found / skipped: $f"
  fi
done

# -------- 4) Benchmarks --------
BENCH_FILES=(
  "aid.zip" "eurosat.zip" "fmow.zip" "millionaid.zip" "nwpu.zip"
  "patternnet.zip" "rsicb256.zip" "SkyScript_cls.zip"
  "roof_shape.zip" "smoothness.zip" "surface.zip"
  "RSICD.zip" "RSITMD.zip" "ucmcaptions.zip"
)

for f in "${BENCH_FILES[@]}"; do
  download "$BASE/benchmark/$f" "$ROOT/benchmark/$f"
done

# -------- 5) Checkpoints --------
CKPT_FILES=(
  "SkyCLIP_ViT_L14_top30pct.zip"
  "SkyCLIP_ViT_L14_top50pct.zip"
  "SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS.zip"
  "SkyCLIP_ViT_L14_top30pct_multi_objects.zip"
  "SkyCLIP_ViT_B32_top50pct.zip"
  "CLIP_ViT_L14_LAION_RS.zip"
)

for f in "${CKPT_FILES[@]}"; do
  download "$BASE/ckpt/$f" "$ROOT/ckpt/$f"
done

echo
echo " DONE. Files are in: $ROOT"
