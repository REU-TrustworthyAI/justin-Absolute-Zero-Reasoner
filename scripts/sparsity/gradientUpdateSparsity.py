# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "torch",
#     "safetensors",
#     "transformers",
#     "accelerate",
# ]
# ///
import os
import safetensors
import torch
import re
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import OrderedDict
from transformers import AutoModelForCausalLM

def calculate_sparsity(state_dict1, state_dict2, threshold=1e-5):
    """
    Calculates the percentage of weights that changed by less than a threshold.

    Args:
        state_dict1 (dict): The first model state dictionary.
        state_dict2 (dict): The second model state dictionary.
        threshold (float): The threshold to consider a weight "untouched".

    Returns:
        float: The percentage of weights that are considered sparse.
    """
    total_params = 0
    untouched_params = 0

    for key in state_dict1:
        if key in state_dict2:
            diff = torch.abs(state_dict1[key].cpu() - state_dict2[key].cpu())
            total_params += diff.numel()
            untouched_params += torch.sum(diff < threshold).item()

    if total_params == 0:
        return 100.0

    sparsity = (untouched_params / total_params) * 100
    return sparsity

def load_model_state_dict(checkpoint_dir):
    """
    Loads a model's state dictionary from a checkpoint directory.
    Handles standard, Hugging Face (sharded or single-file), and FSDP sharded checkpoints.

    Args:
        checkpoint_dir (Path): The directory of a single checkpoint step.

    Returns:
        dict or None: The model's state dictionary if found, otherwise None.
    """
    if not checkpoint_dir.is_dir():
        return None

    if (checkpoint_dir / 'config.json').exists():
        print(f"  -> Found Hugging Face model at '{checkpoint_dir}'. Loading...")
        try:
            # Use from_pretrained to handle sharded and single-file models automatically
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            return model.state_dict()
        except Exception as e:
            print(f"    Error loading Hugging Face model: {e}")

    return None


def plot_sparsity_graph(checkpoint_folder, output_dir, compare_to_initial, base_model_path):
    """
    Loads checkpoints, calculates gradient update sparsity, and plots the results.

    Args:
        checkpoint_folder (str): The path to the folder containing the checkpoints.
        output_dir (str): The directory where the output graph will be saved.
        compare_to_initial (bool): If True, compare each checkpoint to the initial one.
        base_model_path (str or None): Path or Hub ID to a base model to compare against.
    """
    p = Path(checkpoint_folder)
    if not p.is_dir():
        print(f"Error: The provided path '{checkpoint_folder}' is not a valid directory.")
        return

    checkpoint_dirs = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.startswith('global_step_')],
        key=lambda x: int(re.search(r'\d+', x.name).group())
    )

    if len(checkpoint_dirs) < 1:
        print("Error: No step checkpoints found to analyze.")
        return
    if len(checkpoint_dirs) < 2 and not base_model_path and not compare_to_initial:
        print("Error: Need at least two checkpoints for consecutive comparison.")
        return

    actor_sparsities = []
    steps = []
    plot_title = ''

    print(f"Found {len(checkpoint_dirs)} checkpoints. Analyzing sparsity...")

    if base_model_path:
        plot_title = 'Sparsity vs. Base Model'
        print(f"Mode: {plot_title}")

        base_actor_sd = None
        base_model_p = Path(base_model_path)

        if base_model_p.is_dir():
            print(f"Loading base model from local path: {base_model_p}")
            base_actor_sd = load_model_state_dict(base_model_p)
        else:
            print(f"Attempting to load base model from Hugging Face Hub: '{base_model_path}'")
            try:
                model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
                base_actor_sd = model.state_dict()
                print("Successfully loaded base model from Hub.")
            except Exception as e:
                print(f"Error loading model from Hugging Face Hub: {e}")
                return

        if not base_actor_sd:
            print("Error: Could not load the base actor model. Aborting.")
            return

        for current_step_dir in checkpoint_dirs:
            step_number = int(re.search(r'\d+', current_step_dir.name).group())
            steps.append(step_number)
            print(f"Comparing base model with {current_step_dir.name}...")

            current_actor_sd = load_model_state_dict(current_step_dir / 'actor/huggingface/new')

            if current_actor_sd:
                actor_sparsity = calculate_sparsity(base_actor_sd, current_actor_sd)
                actor_sparsities.append(actor_sparsity)
            else:
                print(f"Warning: Could not find actor model for step {step_number}.")
                actor_sparsities.append(None)

    elif compare_to_initial:
        plot_title = 'Sparsity vs. Initial Model'
        print(f"Mode: {plot_title}")
        initial_step_dir = checkpoint_dirs[0]
        print(f"Loading initial model from: {initial_step_dir.name}")

        initial_actor_sd = load_model_state_dict(initial_step_dir / 'actor/huggingface/new')

        if not initial_actor_sd:
            print("Error: Could not load the initial actor model. Aborting.")
            return

        for i in range(1, len(checkpoint_dirs)):
            current_step_dir = checkpoint_dirs[i]
            step_number = int(re.search(r'\d+', current_step_dir.name).group())
            steps.append(step_number)
            print(f"Comparing {initial_step_dir.name} and {current_step_dir.name}...")

            current_actor_sd = load_model_state_dict(current_step_dir / 'actor')

            if current_actor_sd:
                actor_sparsity = calculate_sparsity(initial_actor_sd, current_actor_sd)
                actor_sparsities.append(actor_sparsity)
            else:
                print(f"Warning: Could not find actor model for step {step_number}.")
                actor_sparsities.append(None)

    else:
        plot_title = 'Sparsity Between Consecutive Checkpoints'
        print(f"Mode: {plot_title}")
        for i in range(len(checkpoint_dirs) - 1):
            step_dir1 = checkpoint_dirs[i]
            step_dir2 = checkpoint_dirs[i+1]

            step_number = int(re.search(r'\d+', step_dir2.name).group())
            steps.append(step_number)

            print(f"Comparing {step_dir1.name} and {step_dir2.name}...")

            actor_sd1 = load_model_state_dict(step_dir1 / 'actor/huggingface/new')
            actor_sd2 = load_model_state_dict(step_dir2 / 'actor/huggingface/new')

            if actor_sd1 and actor_sd2:
                actor_sparsity = calculate_sparsity(actor_sd1, actor_sd2)
                actor_sparsities.append(actor_sparsity)
            else:
                print(f"Warning: Could not find actor models for step {step_number}.")
                actor_sparsities.append(None)

    # Plotting
    plt.figure(figsize=(12, 7))

    valid_actor_steps = [steps[i] for i, v in enumerate(actor_sparsities) if v is not None]
    valid_actor_sparsities = [v for v in actor_sparsities if v is not None]

    if valid_actor_sparsities:
        plt.plot(valid_actor_steps, valid_actor_sparsities, marker='o', linestyle='-', label='Actor Sparsity')

    plt.title(plot_title)
    plt.xlabel('Training Step')
    plt.ylabel('Sparsity (%)')
    plt.grid(True)
    plt.legend()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    graph_file = output_path / "gradient_sparsity_graph.png"
    plt.savefig(graph_file)
    print(f"\nGraph saved to {graph_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and plot the gradient update sparsity between checkpoints.")
    parser.add_argument("--checkpoint_folder", type=str, help="The path to the folder containing the checkpoint directories.")
    parser.add_argument("--output-dir", type=str, default=".", help="The directory to save the output graph. Defaults to the current directory.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compare-to-initial", action="store_true", help="Compare each checkpoint to the initial model instead of the previous one.")
    group.add_argument("--base-model-path", type=str, default=None, help="Path or Hugging Face Hub ID of a base model to compare all checkpoints against.")

    args = parser.parse_args()

    plot_sparsity_graph(args.checkpoint_folder, args.output_dir, args.compare_to_initial, args.base_model_path)

