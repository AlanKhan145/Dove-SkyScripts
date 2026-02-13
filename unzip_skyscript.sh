#!/usr/bin/env bash
# unzip_skyscript.sh
# Unzip SkyScript downloads into folder structure expected by repo,
# and auto-fix the "folder inside same-named folder" issue.
#
# Usage:
#   bash unzip_skyscript.sh
#
# Optional env:
#   SKYSCRIPT_ROOT=/home/hungvu/code/khanh/SkyScript/data bash unzip_skyscript.sh
#   UNZIP_IMAGES=0 bash unzip_skyscript.sh
#   UNZIP_META=0 bash unzip_skyscript.sh
#   UNZIP_BENCH=0 bash unzip_skyscript.sh
#   UNZIP_CKPT=0 bash unzip_skyscript.sh
#   FORCE=1 bash unzip_skyscript.sh   # overwrite existing files
#   REPAIR_ONLY=1 bash unzip_skyscript.sh  # chỉ sửa nesting, không unzip

set -euo pipefail

SKYSCRIPT_ROOT="${SKYSCRIPT_ROOT:-/home/hungvu/code/khanh/SkyScript/data}"

UNZIP_IMAGES="${UNZIP_IMAGES:-1}"
UNZIP_META="${UNZIP_META:-1}"
UNZIP_BENCH="${UNZIP_BENCH:-1}"
UNZIP_CKPT="${UNZIP_CKPT:-1}"
FORCE="${FORCE:-0}"
REPAIR_ONLY="${REPAIR_ONLY:-0}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1"; exit 1; }
}
need_cmd unzip
need_cmd find
need_cmd mktemp

log() { echo -e "\n==> $*"; }

# Return 0 if dir has "meaningful" entries (ignore __MACOSX, .DS_Store, Thumbs.db), else 1.
dir_has_content() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  local n
  n="$(find "$d" -mindepth 1 -maxdepth 1 \
      ! -name '__MACOSX' ! -name '.DS_Store' ! -name 'Thumbs.db' \
      -printf '.' 2>/dev/null | wc -c)"
  [[ "$n" -gt 0 ]]
}

# Collapse pattern:
#   outdir/<same-name>/...  => outdir/...
# Safe + fast because it renames/moves the inner directory as a whole.
collapse_double_dir() {
  local outdir="$1"
  local base="${2:-$(basename "$outdir")}"

  [[ -d "$outdir" ]] || return 0

  # Try multiple times in case it is nested more than once
  for _ in 1 2 3; do
    # List top-level entries excluding common junk
    mapfile -t entries < <(
      find "$outdir" -mindepth 1 -maxdepth 1 \
        ! -name '__MACOSX' ! -name '.DS_Store' ! -name 'Thumbs.db' \
        -printf '%f\n' 2>/dev/null
    )

    # If not exactly one entry, nothing to collapse
    [[ "${#entries[@]}" -eq 1 ]] || return 0

    local inner="${entries[0]}"
    local outbase
    outbase="$(basename "$outdir")"

    # Only collapse when inner folder equals expected names
    if [[ "$inner" != "$outbase" && "$inner" != "$base" ]]; then
      return 0
    fi

    [[ -d "$outdir/$inner" ]] || return 0

    log "Fix nesting: $outdir/$inner -> $outdir"

    local parent tmpdir tmpinner
    parent="$(dirname "$outdir")"
    tmpdir="$(mktemp -d "$parent/.tmp_${outbase}_XXXXXX")"
    tmpinner="$tmpdir/$inner"

    # Move the inner directory out, remove wrapper, then move back as outdir
    mv "$outdir/$inner" "$tmpinner"
    rm -rf "$outdir"
    mv "$tmpinner" "$outdir"
    rmdir "$tmpdir" 2>/dev/null || true
  done
}

unzip_one() {
  local zipfile="$1"
  local outdir="$2"
  local base="${3:-$(basename "$zipfile" .zip)}"

  if [[ ! -f "$zipfile" ]]; then
    echo "SKIP (not found): $zipfile"
    # Still try to repair if folder already exists
    collapse_double_dir "$outdir" "$base"
    return 0
  fi

  mkdir -p "$outdir"

  # Repair existing nesting before any decision to skip/unzip
  collapse_double_dir "$outdir" "$base"

  # If only repairing, stop here
  if [[ "$REPAIR_ONLY" == "1" ]]; then
    echo "REPAIR_ONLY=1 -> no unzip: $(basename "$zipfile")"
    return 0
  fi

  # If outdir already has content and not FORCE, skip unzip (but we already repaired)
  if [[ "$FORCE" != "1" ]] && dir_has_content "$outdir"; then
    echo "SKIP (already extracted): $(basename "$zipfile") -> $outdir"
    return 0
  fi

  echo "Unzipping: $(basename "$zipfile") -> $outdir"
  if [[ "$FORCE" == "1" ]]; then
    unzip -q -o "$zipfile" -d "$outdir"
  else
    unzip -q "$zipfile" -d "$outdir"
  fi

  # Repair nesting after unzip
  collapse_double_dir "$outdir" "$base"
}

log "Root: $SKYSCRIPT_ROOT"
echo "UNZIP_IMAGES=$UNZIP_IMAGES UNZIP_META=$UNZIP_META UNZIP_BENCH=$UNZIP_BENCH UNZIP_CKPT=$UNZIP_CKPT FORCE=$FORCE REPAIR_ONLY=$REPAIR_ONLY"

# -------- 1) Images: data/images/images2.zip..images7.zip -> data/images/images2/..images7/ --------
if [[ "$UNZIP_IMAGES" == "1" ]]; then
  log "Unzipping IMAGES..."
  for i in {2..7}; do
    unzip_one "$SKYSCRIPT_ROOT/images/images${i}.zip" \
              "$SKYSCRIPT_ROOT/images/images${i}" \
              "images${i}"
  done
else
  log "Skip IMAGES (UNZIP_IMAGES=0)"
fi

# -------- 2) Meta: data/meta/meta2.zip..meta7.zip -> data/meta/meta2/..meta7/ --------
if [[ "$UNZIP_META" == "1" ]]; then
  log "Unzipping META..."
  for i in {2..7}; do
    unzip_one "$SKYSCRIPT_ROOT/meta/meta${i}.zip" \
              "$SKYSCRIPT_ROOT/meta/meta${i}" \
              "meta${i}"
  done
else
  log "Skip META (UNZIP_META=0)"
fi

# -------- 3) Benchmarks: data/benchmark/*.zip -> data/benchmark/<zip_name_without_ext>/ --------
if [[ "$UNZIP_BENCH" == "1" ]]; then
  log "Unzipping BENCHMARK datasets..."
  shopt -s nullglob
  for z in "$SKYSCRIPT_ROOT/benchmark/"*.zip; do
    base="$(basename "$z" .zip)"
    unzip_one "$z" "$SKYSCRIPT_ROOT/benchmark/$base" "$base"
  done
  shopt -u nullglob
else
  log "Skip BENCH (UNZIP_BENCH=0)"
fi

# -------- 4) Checkpoints: data/ckpt/*.zip -> data/ckpt/<zip_name_without_ext>/ --------
if [[ "$UNZIP_CKPT" == "1" ]]; then
  log "Unzipping MODEL CHECKPOINTS..."
  shopt -s nullglob
  for z in "$SKYSCRIPT_ROOT/ckpt/"*.zip; do
    base="$(basename "$z" .zip)"
    unzip_one "$z" "$SKYSCRIPT_ROOT/ckpt/$base" "$base"
  done
  shopt -u nullglob
else
  log "Skip CKPT (UNZIP_CKPT=0)"
fi

log "Done."
echo "Tip: find checkpoint files after unzip:"
echo "  find \"$SKYSCRIPT_ROOT/ckpt\" -type f | head -n 30"
