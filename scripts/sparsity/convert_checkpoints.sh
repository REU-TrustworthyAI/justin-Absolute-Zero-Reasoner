#!/bin/bash

# A script to iterate through veRL checkpoint directories and convert
# the sharded actor models to the Hugging Face safetensors format.

# --- Usage ---
# ./convert_checkpoints.sh <path_to_main_checkpoints_folder> <path_to_hf_base_model>
#
# Example:
# ./convert_checkpoints.sh ./my_experiment_checkpoints/ ./base_model/

# --- Script Start ---

# 1. Argument Validation
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_main_checkpoints_folder>"
    exit 1
fi

CHECKPOINTS_DIR=$1

# Check if the provided paths are valid directories
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Error: Checkpoints directory not found at '$CHECKPOINTS_DIR'"
    exit 1
fi

# 2. Iterate and Convert
# Find all directories that start with "global_step_"
for step_dir in "$CHECKPOINTS_DIR"/global_step_*; do
    # Check if it's actually a directory
    if [ -d "$step_dir" ]; then
        echo "--------------------------------------------------"
        echo "Processing checkpoint: $step_dir"
        echo "--------------------------------------------------"

        # Define the paths required by the conversion script
        verl_actor_path="$step_dir/actor"
        hf_path="$step_dir/actor/huggingface"

        # Ensure the actor directory exists before trying to convert
        if [ ! -d "$verl_actor_path" ]; then
            echo "Warning: Actor directory not found in '$step_dir'. Skipping."
            continue
        fi

        # Run the conversion command
        echo "Running conversion script..."
        python -m absolute_zero_reasoner.utils.convert2hf \
            "$verl_actor_path" \
            "$hf_path" \
            "$hf_path/new" \
	    --world_size=2

        if [ $? -eq 0 ]; then
            echo "Successfully converted checkpoint in '$step_dir'."
        else
            echo "Error during conversion for '$step_dir'."
        fi
        echo ""
    fi
done

echo "All checkpoint conversions complete."
