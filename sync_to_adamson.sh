#!/bin/bash
#
# sync_to_adamson.sh
#
# DESCRIPTION:
#   Converts main code/scripts to Adamson versions with proper file replacements.
#   This script automatically syncs:
#     1. code/ → code_Adamson/
#     2. slurm_scripts/ → slurm_scripts_Adamson/
#
# WHAT IT DOES:
#   - Copies all Python scripts from code/ to code_Adamson/
#   - Copies all SLURM scripts from slurm_scripts/ to slurm_scripts_Adamson/
#   - Applies 13 find-and-replace patterns for Adamson-specific naming
#   - Preserves Adamson-specific files (doesn't overwrite them)
#
# USAGE:
#   bash sync_to_adamson.sh           # Normal run
#   bash sync_to_adamson.sh --help    # Show this help
#
# LOCATION:
#   Must be run from project root directory
#

set -e # Exit on error

# Show help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
	head -n 25 "$0" | tail -n 20
	exit 0
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory (script location)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SLMGAE → Adamson Conversion Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "This script will sync:"
echo "  • code/ → code_Adamson/"
echo "  • slurm_scripts/ → slurm_scripts_Adamson/"
echo ""

# ============================================================================
# Pre-flight checks
# ============================================================================
echo -e "${YELLOW}Pre-flight checks...${NC}"

if [ ! -d "code" ]; then
	echo -e "${RED}✗ code/ directory not found${NC}"
	echo "Please run this script from the project root directory"
	exit 1
fi

if [ ! -d "slurm_scripts" ]; then
	echo -e "${RED}✗ slurm_scripts/ directory not found${NC}"
	echo "Please run this script from the project root directory"
	exit 1
fi

echo -e "${GREEN}✓ Source directories found${NC}"
echo ""

# ============================================================================
# PART 1: Sync code/ to code_Adamson/
# ============================================================================

echo -e "${YELLOW}[1/2] Syncing code/ to code_Adamson/${NC}"
echo ""

# List of files to sync (exclude Adamson-specific files)
CODE_FILES=(
	"slmgae_pytorch.py"
	"train_slmgae.py"
	"train_slmgae_with_esm.py"
	"train_bc.py"
	"case_study.py"
	"data_split.py"
	"preprocess_benchmarking_paper.py"
	"objective.py"
	"evaluation.py"
	"fetch_all_proteins.py"
	"generate_esm_embeddings_gpu.py"
	"gene_corrections_config.py"
)

# Files that should NOT be synced (Adamson-specific)
ADAMSON_SPECIFIC=(
	"create_adamson_sl_file.py"
	"visualize_sl_distribution_Adamson.py"
	"adamson_sl_distribution.png"
	"REFACTORING_README.md"
	"CLAUDE.md"
)

echo "Files to sync: ${#CODE_FILES[@]}"
echo "Adamson-specific files (will be preserved): ${#ADAMSON_SPECIFIC[@]}"
echo ""

# Create code_Adamson if it doesn't exist
mkdir -p code_Adamson

# Sync each file with replacements
for FILE in "${CODE_FILES[@]}"; do
	if [ -f "code/$FILE" ]; then
		echo -e "  Processing: ${GREEN}$FILE${NC}"

		# Read file and apply replacements
		sed \
			-e 's|SL_Human_Approved\.txt|SL_Adamson_gamma_thresholding.txt|g' \
			-e 's|fold_\([0-9]*\)_best\.pt|fold_\1_best_Adamson.pt|g' \
			-e 's|fold_\([0-9]*\)_predictions\.csv|fold_\1_predictions_Adamson.csv|g' \
			-e 's|fold_{fold_idx}_best\.pt|fold_{fold_idx}_best_Adamson.pt|g' \
			-e 's|fold_{fold_idx}_predictions\.csv|fold_{fold_idx}_predictions_Adamson.csv|g' \
			-e 's|fold_{fold}_best\.pt|fold_{fold}_best_Adamson.pt|g' \
			-e 's|fold_{fold}_predictions\.csv|fold_{fold}_predictions_Adamson.csv|g' \
			-e 's|training_summary\.json|training_summary_Adamson.json|g' \
			-e 's|training_log\.json|training_log_Adamson.json|g' \
			-e 's|results/slmgae_pytorch|../outputs_Adamson/main_without_ESM|g' \
			-e 's|results/slmgae_with_esm|../outputs_Adamson/main_with_ESM|g' \
			-e 's|results/bc_without_esm|../outputs_Adamson/BC_without_ESM|g' \
			-e 's|results/case_study|../outputs_Adamson/case_study_without_ESM|g' \
			"code/$FILE" >"code_Adamson/$FILE"

		echo -e "    ${GREEN}✓${NC} Copied with replacements"
	else
		echo -e "  ${RED}✗${NC} Missing: code/$FILE"
	fi
done

echo ""
echo -e "${GREEN}✓${NC} Code sync complete"
echo ""

# ============================================================================
# PART 2: Sync slurm_scripts/ to slurm_scripts_Adamson/
# ============================================================================

echo -e "${YELLOW}[2/2] Syncing slurm_scripts/ to slurm_scripts_Adamson/${NC}"
echo ""

# List of SLURM scripts to sync
SLURM_FILES=(
	"step1_download_protein_seq.sh"
	"step2_generate_ESM_embedding.sh"
	"step3_train_main_without_ESM.sh"
	"step3_train_main_with_ESM.sh"
	"step3_train_CV2.sh"
	"step3_train_CV3.sh"
	"step4_train_BC_without_ESM.sh"
	"step5_case_study.sh"
	"run_all_pipeline.sh"
	"tmux_pipeline.sh"
)

# Create slurm_scripts_Adamson if it doesn't exist
mkdir -p slurm_scripts_Adamson

# Sync each SLURM script with replacements
for FILE in "${SLURM_FILES[@]}"; do
	if [ -f "slurm_scripts/$FILE" ]; then
		echo -e "  Processing: ${GREEN}$FILE${NC}"

		# Read file and apply replacements
		sed \
			-e 's|#SBATCH --job-name=slmgae|#SBATCH --job-name=slmgae_adamson|g' \
			-e 's|#SBATCH --job-name=SLMGAE|#SBATCH --job-name=slmgae_adamson|g' \
			-e 's|#SBATCH --output=slmgae|#SBATCH --output=slmgae_adamson|g' \
			-e 's|#SBATCH --output=outputs/|#SBATCH --output=outputs_Adamson/|g' \
			-e 's|cd \.\./code|cd ../code_Adamson|g' \
			-e 's|cd code|cd code_Adamson|g' \
			-e 's|--output_dir outputs/|--output_dir outputs_Adamson/|g' \
			-e 's|--output_dir ../outputs/|--output_dir ../outputs_Adamson/|g' \
			-e 's|mkdir -p outputs/|mkdir -p outputs_Adamson/|g' \
			-e 's|mkdir -p ../outputs/|mkdir -p ../outputs_Adamson/|g' \
			-e 's|\.\./outputs/|../outputs_Adamson/|g' \
			"slurm_scripts/$FILE" >"slurm_scripts_Adamson/$FILE"

		# Preserve execute permissions
		chmod +x "slurm_scripts_Adamson/$FILE"

		echo -e "    ${GREEN}✓${NC} Copied with replacements"
	else
		echo -e "  ${RED}✗${NC} Missing: slurm_scripts/$FILE"
	fi
done

echo ""
echo -e "${GREEN}✓${NC} SLURM scripts sync complete"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Sync Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Files synced:"
echo "  • ${#CODE_FILES[@]} Python scripts (code/ → code_Adamson/)"
echo "  • ${#SLURM_FILES[@]} SLURM scripts (slurm_scripts/ → slurm_scripts_Adamson/)"
echo ""
echo "Find-and-replace patterns applied (13 total):"
echo ""
echo "Data files:"
echo "  • SL_Human_Approved.txt → SL_Adamson_gamma_thresholding.txt"
echo ""
echo "Output files (f-strings handled):"
echo "  • fold_X_best.pt → fold_X_best_Adamson.pt"
echo "  • fold_{fold_idx}_best.pt → fold_{fold_idx}_best_Adamson.pt"
echo "  • fold_X_predictions.csv → fold_X_predictions_Adamson.csv"
echo "  • fold_{fold_idx}_predictions.csv → fold_{fold_idx}_predictions_Adamson.csv"
echo "  • training_summary.json → training_summary_Adamson.json"
echo "  • training_log.json → training_log_Adamson.json"
echo ""
echo "Directories:"
echo "  • code/ → code_Adamson/"
echo "  • outputs/ → outputs_Adamson/"
echo "  • slurm_scripts/ → slurm_scripts_Adamson/"
echo ""
echo "SLURM job names:"
echo "  • slmgae → slmgae_adamson"
echo ""
echo -e "${YELLOW}Preserved (not overwritten):${NC}"
for FILE in "${ADAMSON_SPECIFIC[@]}"; do
	echo "  • $FILE"
done
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Review changes: git diff code_Adamson/ slurm_scripts_Adamson/"
echo "  2. Run Adamson pipeline: bash slurm_scripts_Adamson/run_all_pipeline.sh"
echo ""
echo -e "${GREEN}✓ Ready to run Adamson version!${NC}"
