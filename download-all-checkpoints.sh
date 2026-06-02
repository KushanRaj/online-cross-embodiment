#!/bin/bash
# =============================================================
# Download ALL model checkpoints for VLA research
# =============================================================
# Models:
#   - Show-o2 1.5B & 7B (base for IVLR architecture)
#   - MolmoAct2 + all fine-tuned variants (5B each)
#   - DreamZero-DROID (23B, ~65GB — skip with --no-dreamzero)
#
# Requirements:
#   pip install huggingface_hub
#   huggingface-cli login  (for gated models, if any)
#
# Usage:
#   bash download-all-checkpoints.sh                # download everything
#   bash download-all-checkpoints.sh --no-dreamzero # skip DreamZero (65GB)
#   bash download-all-checkpoints.sh --dry-run      # just show what would download
# =============================================================

set -e

# --- Configuration ---
BASE_DIR="${CHECKPOINT_DIR:-$HOME/vla-checkpoints}"
SKIP_DREAMZERO=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --no-dreamzero) SKIP_DREAMZERO=true ;;
        --dry-run) DRY_RUN=true ;;
        --dir=*) BASE_DIR="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# --- Helper ---
download_model() {
    local repo=$1
    local name=$2
    local dest="$BASE_DIR/$name"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📦 $name"
    echo "  Repo: $repo"
    echo "  Dest: $dest"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -d "$dest" ] && [ "$(ls -A $dest 2>/dev/null)" ]; then
        echo "  ✅ Already exists, skipping. Delete folder to re-download."
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would download to $dest"
        return 0
    fi

    mkdir -p "$dest"
    huggingface-cli download "$repo" --local-dir "$dest"
    echo "  ✅ Done: $name"
}

# --- Pre-flight ---
echo "============================================="
echo "  VLA Checkpoint Downloader"
echo "============================================="
echo "  Download directory: $BASE_DIR"
echo "  Skip DreamZero:    $SKIP_DREAMZERO"
echo "  Dry run:           $DRY_RUN"
echo ""

if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found."
    echo "Install with: pip install huggingface_hub"
    echo "Then login:   huggingface-cli login"
    exit 1
fi

mkdir -p "$BASE_DIR"

# --- Estimated sizes ---
echo "Estimated total download sizes:"
echo "  Show-o2 1.5B:              ~3 GB"
echo "  Show-o2 7B:                ~14 GB"
echo "  MolmoAct2 (base):          ~20 GB (F32) / ~10 GB (BF16)"
echo "  MolmoAct2-LIBERO:          ~10 GB"
echo "  MolmoAct2-DROID:           ~10 GB"
echo "  MolmoAct2-BimanualYAM:     ~10 GB"
echo "  MolmoAct2-SO100_101:       ~10 GB"
echo "  DreamZero-DROID:           ~65 GB"
echo "  ─────────────────────────────────"
if [ "$SKIP_DREAMZERO" = true ]; then
    echo "  Total (no DreamZero):   ~77 GB"
else
    echo "  Total:                  ~142 GB"
fi
echo ""

# ============================================================
# 1. Show-o2 (base models for IVLR-style architecture)
# ============================================================
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  SECTION 1: Show-o2 base models           ║"
echo "╚════════════════════════════════════════════╝"

download_model "showlab/show-o2-1.5B" "show-o2-1.5B"
download_model "showlab/show-o2-7B"   "show-o2-7B"

# ============================================================
# 2. MolmoAct2 — all variants
# ============================================================
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  SECTION 2: MolmoAct2 + fine-tuned variants║"
echo "╚════════════════════════════════════════════╝"

# Base model (general-purpose VLA, 5B params)
download_model "allenai/MolmoAct2" "MolmoAct2"

# Fine-tuned on LIBERO benchmark (130 Franka tasks)
download_model "allenai/MolmoAct2-LIBERO" "MolmoAct2-LIBERO"

# Fine-tuned on DROID dataset (distributed robot interaction)
download_model "allenai/MolmoAct2-DROID" "MolmoAct2-DROID"

# Fine-tuned on bimanual YAM data
download_model "allenai/MolmoAct2-BimanualYAM" "MolmoAct2-BimanualYAM"

# Fine-tuned on SO-100/101 real hardware data
download_model "allenai/MolmoAct2-SO100_101" "MolmoAct2-SO100_101"

# ============================================================
# 3. DreamZero (23B world action model)
# ============================================================
if [ "$SKIP_DREAMZERO" = false ]; then
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║  SECTION 3: DreamZero-DROID (23B, ~65GB)  ║"
    echo "╚════════════════════════════════════════════╝"
    echo "  ⚠️  This is a large download (~65GB)."
    echo "  Run with --no-dreamzero to skip."

    download_model "GEAR-Dreams/DreamZero-DROID" "DreamZero-DROID"
else
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║  SECTION 3: DreamZero — SKIPPED           ║"
    echo "╚════════════════════════════════════════════╝"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================="
echo "  DOWNLOAD COMPLETE"
echo "============================================="
echo ""
echo "All checkpoints saved to: $BASE_DIR"
echo ""
echo "Directory structure:"
ls -1d "$BASE_DIR"/*/ 2>/dev/null | while read dir; do
    name=$(basename "$dir")
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "  $name ($size)"
done
echo ""
echo "To use a different directory next time:"
echo "  CHECKPOINT_DIR=/path/to/dir bash download-all-checkpoints.sh"
echo ""
echo "Next steps:"
echo "  - MolmoAct2 (5B): runs on 1x L4 24GB (AWS g6.2xlarge ~\$0.98/hr)"
echo "  - DreamZero (23B): needs 4x A10G (AWS g5.12xlarge ~\$5.67/hr)"
echo "  - Show-o2 7B: runs on 1x L4 24GB"
echo "  - Show-o2 1.5B: runs on any GPU with 8GB+ VRAM"
