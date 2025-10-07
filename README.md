# SL_GNN

## Graph Neural Network for Synthetic Lethality Prediction

### Project Structure

- **Code**: Located in the `code/` directory
- **Code Sync**: Use `sync_to_adamson.sh` to duplicate code for Adamson data
- **Slurm Scripts**: Available in the `slurm_scripts/` directory

### Pipeline Execution

Run `tmux_pipeline.sh` in a tmux window to execute the following workflow:

1. Download Amino Acid information with corrected names
2. Generate protein embeddings for each gene using mean pooling with ESM embeddings
3. Fit SLMGAE model (with or without ESM embeddings)

### Training Details

The main SLMGAE model is trained on the **SLDB**. 

> **Note**: BC and case study datasets are not currently of interest.