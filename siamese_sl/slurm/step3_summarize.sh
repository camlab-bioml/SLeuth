#!/bin/bash
#SBATCH --job-name=siamese_summary
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm/logs/step3_summary_%j.out
#SBATCH --error=slurm/logs/step3_summary_%j.err

echo "=================================================="
echo "Siamese SL - Results Summary"
echo "=================================================="
echo "Time: $(date)"
echo "=================================================="

# Change to siamese_sl directory (where sbatch was called from)
cd "$SLURM_SUBMIT_DIR"

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/pytorch/bin/python

# Generate summary
$PYTHON_PATH -c "
import json
from pathlib import Path
from datetime import datetime

print()
print('=' * 70)
print('SIAMESE SL - RKHS-BASED SYNTHETIC LETHALITY PREDICTION')
print('=' * 70)
print()

results_dir = Path('results')
summary_data = {}

for cv in ['cv1', 'cv2', 'cv3']:
    results_file = results_dir / cv / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)

        summary = data.get('summary', {})
        cv_desc = data.get('cv_description', cv.upper())

        print(f'{cv_desc}')
        print('-' * 40)
        print(f\"  AUROC: {summary.get('auroc_mean', 0):.4f} ± {summary.get('auroc_std', 0):.4f}\")
        print(f\"  AUPR:  {summary.get('aupr_mean', 0):.4f} ± {summary.get('aupr_std', 0):.4f}\")
        print(f\"  F1:    {summary.get('f1_mean', 0):.4f} ± {summary.get('f1_std', 0):.4f}\")
        print()

        summary_data[cv] = summary
    else:
        print(f'{cv.upper()}: Results not found')
        print()

# Create comparison table
print('=' * 70)
print('COMPARISON TABLE')
print('=' * 70)
print()
print(f\"{'CV Type':<25} {'AUROC':>12} {'AUPR':>12} {'F1':>12}\")
print('-' * 65)

cv_names = {
    'cv1': 'CV1 (edge-based)',
    'cv2': 'CV2 (gene-based)',
    'cv3': 'CV3 (pair-based)',
}

for cv in ['cv1', 'cv2', 'cv3']:
    if cv in summary_data:
        s = summary_data[cv]
        auroc = f\"{s.get('auroc_mean', 0):.4f}±{s.get('auroc_std', 0):.4f}\"
        aupr = f\"{s.get('aupr_mean', 0):.4f}±{s.get('aupr_std', 0):.4f}\"
        f1 = f\"{s.get('f1_mean', 0):.4f}±{s.get('f1_std', 0):.4f}\"
        print(f\"{cv_names[cv]:<25} {auroc:>12} {aupr:>12} {f1:>12}\")
    else:
        print(f\"{cv_names[cv]:<25} {'N/A':>12} {'N/A':>12} {'N/A':>12}\")

print()
print('=' * 70)
print(f'Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print('=' * 70)

# Save summary to file
with open(results_dir / 'summary.txt', 'w') as f:
    f.write('SIAMESE SL - RKHS-BASED SYNTHETIC LETHALITY PREDICTION\\n')
    f.write('=' * 70 + '\\n\\n')
    f.write(f\"{'CV Type':<25} {'AUROC':>12} {'AUPR':>12} {'F1':>12}\\n\")
    f.write('-' * 65 + '\\n')
    for cv in ['cv1', 'cv2', 'cv3']:
        if cv in summary_data:
            s = summary_data[cv]
            auroc = f\"{s.get('auroc_mean', 0):.4f}±{s.get('auroc_std', 0):.4f}\"
            aupr = f\"{s.get('aupr_mean', 0):.4f}±{s.get('aupr_std', 0):.4f}\"
            f1 = f\"{s.get('f1_mean', 0):.4f}±{s.get('f1_std', 0):.4f}\"
            f.write(f\"{cv_names[cv]:<25} {auroc:>12} {aupr:>12} {f1:>12}\\n\")
    f.write('\\n')
    f.write(f'Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\\n')

print()
print('Summary saved to: results/summary.txt')
"

echo "=================================================="
echo "Summary completed at: $(date)"
echo "=================================================="
