#!/bin/bash
# Sequential SLURM pipeline runner for tmux
# Usage: Run this in a tmux session: bash tmux_pipeline.sh

set -euo pipefail # Exit on error, undefined variables, and pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
POLL_INTERVAL=10 # seconds between status checks
SACCT_WAIT=2     # seconds to wait for sacct to update
MAX_RETRIES=3    # max retries for sacct queries

# Function to print colored messages
print_status() {
	echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
	echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

print_info() {
	echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
	echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to check if a command exists
command_exists() {
	command -v "$1" >/dev/null 2>&1
}

# Verify required commands
check_prerequisites() {
	local missing_cmds=()

	for cmd in sbatch squeue sacct; do
		if ! command_exists "$cmd"; then
			missing_cmds+=("$cmd")
		fi
	done

	if [ ${#missing_cmds[@]} -gt 0 ]; then
		print_error "Missing required commands: ${missing_cmds[*]}"
		print_error "Please ensure you're on a SLURM cluster node"
		exit 1
	fi
}

# Function to wait for a SLURM job to complete
wait_for_job() {
	local job_id=$1
	local job_name=$2
	local start_time=$(date +%s)

	# Validate job_id is numeric
	if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
		print_error "Invalid job ID: $job_id"
		return 1
	fi

	print_info "Waiting for job $job_id ($job_name) to complete..."

	while true; do
		# Check job status with squeue
		local job_status
		job_status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null | head -1)

		if [ -z "$job_status" ]; then
			# Job not in queue, check sacct for final status
			sleep "$SACCT_WAIT"

			local retry_count=0
			local sacct_output=""

			while [ $retry_count -lt $MAX_RETRIES ]; do
				sacct_output=$(sacct -j "$job_id" -n --format=State%20 2>/dev/null | tail -1 | xargs)

				if [ -n "$sacct_output" ]; then
					break
				fi

				((retry_count++))
				sleep "$SACCT_WAIT"
			done

			if [ -z "$sacct_output" ]; then
				print_warning "Could not get job status from sacct after $MAX_RETRIES retries"
				return 1
			fi

			# Clear the running status line
			echo -ne "\033[2K\r"

			case "$sacct_output" in
			*COMPLETED*)
				local elapsed=$(($(date +%s) - start_time))
				print_status "✓ Job $job_id ($job_name) completed successfully! (elapsed: ${elapsed}s)"
				return 0
				;;
			*FAILED* | *TIMEOUT* | *NODE_FAIL* | *OUT_OF_MEMORY*)
				print_error "✗ Job $job_id ($job_name) failed with status: $sacct_output"
				return 1
				;;
			*CANCELLED*)
				print_warning "Job $job_id ($job_name) was cancelled"
				return 1
				;;
			*)
				print_warning "Job $job_id ($job_name) ended with unexpected status: $sacct_output"
				return 1
				;;
			esac
		else
			# Job still in queue/running
			local elapsed=$(($(date +%s) - start_time))
			case "$job_status" in
			RUNNING)
				echo -ne "\033[2K\r${YELLOW}→${NC} Job $job_id: RUNNING... (elapsed: ${elapsed}s, checking in ${POLL_INTERVAL}s)"
				;;
			PENDING)
				echo -ne "\033[2K\r${YELLOW}⏳${NC} Job $job_id: PENDING... (elapsed: ${elapsed}s, checking in ${POLL_INTERVAL}s)"
				;;
			*)
				echo -ne "\033[2K\r${YELLOW}?${NC} Job $job_id: $job_status... (elapsed: ${elapsed}s)"
				;;
			esac
		fi

		sleep "$POLL_INTERVAL"
	done
}

# Function to submit a job and get its ID
submit_job() {
	local script_name=$1

	# Check if script exists and is readable
	if [ ! -f "$script_name" ]; then
		print_error "Script not found: $script_name"
		return 1
	fi

	if [ ! -r "$script_name" ]; then
		print_error "Script not readable: $script_name"
		return 1
	fi

	# Submit job and capture output
	local job_output
	job_output=$(sbatch "$script_name" 2>&1)
	local submit_status=$?

	if [ $submit_status -ne 0 ]; then
		print_error "Failed to submit $script_name: $job_output"
		return 1
	fi

	# Extract job ID from output
	if [[ $job_output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
		echo "${BASH_REMATCH[1]}"
		return 0
	else
		print_error "Could not parse job ID from: $job_output"
		return 1
	fi
}

# Function to create log directory if needed
ensure_log_directory() {
	if [ ! -d "logs" ]; then
		print_info "Creating logs directory..."
		mkdir -p logs || {
			print_error "Failed to create logs directory"
			exit 1
		}
	fi
}

# Main pipeline
main() {
	echo "============================================"
	echo "     SLMGAE Pipeline Runner for TMUX"
	echo "============================================"
	print_status "Starting pipeline at $(date)"
	echo ""

	# Check prerequisites
	check_prerequisites

	# Ensure log directory exists
	ensure_log_directory

	# Track overall pipeline status
	local pipeline_success=true

	# Step 1: Download protein sequences
	print_status "Step 1: Submitting protein download job..."
	local job1_id
	if job1_id=$(submit_job "step1_download_protein_seq.sh"); then
		print_info "Submitted step1_download_protein_seq.sh with job ID: $job1_id"

		if ! wait_for_job "$job1_id" "Protein Download"; then
			print_error "Step 1 failed. Stopping pipeline."
			exit 1
		fi
	else
		print_error "Failed to submit Step 1"
		exit 1
	fi

	echo ""

	# Step 2: Generate ESM embeddings
	print_status "Step 2: Submitting ESM embedding generation job..."
	local job2_id
	if job2_id=$(submit_job "step2_generate_ESM_embedding.sh"); then
		print_info "Submitted step2_generate_ESM_embedding.sh with job ID: $job2_id"

		if ! wait_for_job "$job2_id" "ESM Embedding Generation"; then
			print_error "Step 2 failed. Stopping pipeline."
			exit 1
		fi
	else
		print_error "Failed to submit Step 2"
		exit 1
	fi

	echo ""

	# Step 3: Train main model (both with and without ESM)
	print_status "Step 3: Submitting main training jobs..."

	# Array to track step 3 jobs
	local -a jobs_step3=()
	local step3_failed=false

	# Submit training with ESM
	if [ -f "step3_train_main_with_ESM.sh" ]; then
		local job3a_id
		if job3a_id=$(submit_job "step3_train_main_with_ESM.sh"); then
			print_info "Submitted step3_train_main_with_ESM.sh with job ID: $job3a_id"
			jobs_step3+=("$job3a_id:Main_with_ESM")
		else
			print_warning "Failed to submit training with ESM"
			step3_failed=true
		fi
	else
		print_warning "step3_train_main_with_ESM.sh not found"
	fi

	# Submit training without ESM
	if [ -f "step3_train_main_without_ESM.sh" ]; then
		local job3b_id
		if job3b_id=$(submit_job "step3_train_main_without_ESM.sh"); then
			print_info "Submitted step3_train_main_without_ESM.sh with job ID: $job3b_id"
			jobs_step3+=("$job3b_id:Main_without_ESM")
		else
			print_warning "Failed to submit training without ESM"
			step3_failed=true
		fi
	else
		print_warning "step3_train_main_without_ESM.sh not found"
	fi

	# Submit CV2 (gene-based cross-validation)
	if [ -f "step3_train_CV2.sh" ]; then
		local job3c_id
		if job3c_id=$(submit_job "step3_train_CV2.sh"); then
			print_info "Submitted step3_train_CV2.sh with job ID: $job3c_id"
			jobs_step3+=("$job3c_id:CV2_gene_based")
		else
			print_warning "Failed to submit CV2 training"
			step3_failed=true
		fi
	else
		print_warning "step3_train_CV2.sh not found"
	fi

	# Submit CV3 (pair-based cross-validation)
	if [ -f "step3_train_CV3.sh" ]; then
		local job3d_id
		if job3d_id=$(submit_job "step3_train_CV3.sh"); then
			print_info "Submitted step3_train_CV3.sh with job ID: $job3d_id"
			jobs_step3+=("$job3d_id:CV3_pair_based")
		else
			print_warning "Failed to submit CV3 training"
			step3_failed=true
		fi
	else
		print_warning "step3_train_CV3.sh not found"
	fi

	# Check if we have any Step 3 jobs to wait for
	if [ ${#jobs_step3[@]} -eq 0 ]; then
		print_error "No Step 3 jobs were submitted successfully"
		pipeline_success=false
	else
		# Wait for all Step 3 jobs
		for job_info in "${jobs_step3[@]}"; do
			IFS=':' read -r job_id job_name <<<"$job_info"
			if ! wait_for_job "$job_id" "$job_name"; then
				print_error "Step 3 job $job_name failed"
				step3_failed=true
			fi
		done
	fi

	echo ""

	# Step 4: BC training without ESM (DISABLED)
	# print_status "Step 4: Submitting BC training without ESM..."
	# local job4_id
	# if [ -f "step4_train_BC_without_ESM.sh" ]; then
	#     if job4_id=$(submit_job "step4_train_BC_without_ESM.sh"); then
	#         print_info "Submitted step4_train_BC_without_ESM.sh with job ID: $job4_id"
	#
	#         if ! wait_for_job "$job4_id" "BC_without_ESM"; then
	#             print_error "Step 4 failed. Continuing with pipeline."
	#             pipeline_success=false
	#         fi
	#     else
	#         print_warning "Failed to submit BC training"
	#     fi
	# else
	#     print_warning "step4_train_BC_without_ESM.sh not found"
	# fi

	echo ""

	# Step 5: Case study without ESM (DISABLED)
	# print_status "Step 5: Submitting case study without ESM..."
	# local job5_id
	# if [ -f "step5_case_study.sh" ]; then
	#     if job5_id=$(submit_job "step5_case_study.sh"); then
	#         print_info "Submitted step5_case_study.sh with job ID: $job5_id"
	#
	#         if ! wait_for_job "$job5_id" "Case_study_without_ESM"; then
	#             print_error "Step 5 failed. Continuing with pipeline."
	#             pipeline_success=false
	#         fi
	#     else
	#         print_warning "Failed to submit case study"
	#     fi
	# else
	#     print_warning "step5_case_study.sh not found"
	# fi

	echo ""

	# Summary
	echo "============================================"
	print_status "Pipeline completed at $(date)"
	echo "============================================"
	echo ""

	if [ "$step3_failed" = true ]; then
		print_warning "Some jobs failed during execution"
		pipeline_success=false
	fi

	print_info "Summary of pipeline execution:"
	echo "  ✓ Step 1: Protein sequences downloaded"
	echo "  ✓ Step 2: ESM embeddings generated"

	if [ ${#jobs_step3[@]} -gt 0 ]; then
		echo "  ✓ Step 3: Main model training jobs submitted (${#jobs_step3[@]} jobs)"
	else
		echo "  ✗ Step 3: No training jobs submitted"
	fi

	echo ""
	print_info "Check output directories for results:"
	echo "  - Proteins: ../protein_seq/"
	echo "  - Embeddings: ../ESM_embedding/"
	echo "  - Training results: ../outputs/"
	echo "  - Log files: logs/"
	echo ""

	if [ "$pipeline_success" = true ]; then
		print_status "Pipeline finished successfully!"
		exit 0
	else
		print_warning "Pipeline completed with some warnings/errors"
		exit 1
	fi
}

# Trap to handle interrupts
cleanup() {
	echo "" # New line to clear any running status
	print_error "Pipeline interrupted by user"
	exit 130
}

trap cleanup INT TERM

# Run main pipeline
main "$@"
