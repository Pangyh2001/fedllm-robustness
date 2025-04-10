#!/usr/bin/env python3
# experiment.py

import os
import argparse
import subprocess
import datetime
import json
from pathlib import Path
from itertools import product

def run_experiment(args_dict, experiment_name):
    """Run a single experiment with the given arguments."""
    # Create command with all arguments
    cmd = ["python", "main.py"]
    for key, value in args_dict.items():
        if value is not None:  # Only add if value is not None
            cmd.extend([f"--{key}", str(value)])
    
    # Print command for logging purposes
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Create output directory specific to this experiment
    output_dir = os.path.join("experiments", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Record experiment settings
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    # Redirect output to log file
    log_file = os.path.join(output_dir, "experiment.log")
    with open(log_file, "w") as f:
        # Run the experiment and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        # Wait for process to complete
        process.wait()
    
    return process.returncode

def main():
    # Timestamp for experiment folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"experiments_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Base configuration that will be used for all experiments
    base_config = {
        "model_name": "google/gemma-1.1-2b-it",
        "max_samples": 2000,
        "max_length": 256,
        "num_clients": 5,
        "client_fraction": 1.0,
        "num_rounds": 20,  # 50 training rounds as specified
        "local_epochs": 1,
        "batch_size": 8,
        "lr": 2e-4,
        "seed": 42,
        "epsilon": 0.05,
        "adv_weight": 0.3,
    }
    
    # Dataset variations
    datasets = ["agnews", "bbcnews","reuters"]  # "imdb"运行过了，不需要再运行了
    
    # Algorithm variations
    algorithms = ["fedavg", "cat", "cat2"]
    
    # CAT2 parameter variations
    batch_thresholds = [7/8, 6/8, 5/8]
    confidence_thresholds = [0.7, 0.8, 0.95]
    
    # Run experiments with different datasets and algorithms
    for dataset in datasets:
        print(f"\n\n{'*'*80}")
        print(f"Starting experiments for dataset: {dataset}")
        print(f"{'*'*80}\n")
        
        for algorithm in algorithms:
            # Basic experiment with current dataset and algorithm
            config = base_config.copy()
            config["dataset"] = dataset
            config["algorithm"] = algorithm
            config["output_dir"] = os.path.join(base_output_dir, f"{dataset}_{algorithm}_base")
            
            # For CAT2, we add the default CAT2 parameters
            if algorithm == "cat2":
                config["confidence_threshold"] = 0.9
                config["batch_threshold"] = 0.9
            
            # Run the basic experiment
            exp_name = f"{dataset}_{algorithm}_base"
            run_experiment(config, exp_name)
            
            # For CAT2, run additional experiments with varied parameters
            if algorithm == "cat2":
                # Vary batch_threshold
                for batch_threshold in batch_thresholds:
                    config = base_config.copy()
                    config["dataset"] = dataset
                    config["algorithm"] = algorithm
                    config["confidence_threshold"] = 0.9  # Default
                    config["batch_threshold"] = batch_threshold
                    config["output_dir"] = os.path.join(base_output_dir, 
                                                       f"{dataset}_{algorithm}_batch{batch_threshold:.2f}")
                    
                    exp_name = f"{dataset}_{algorithm}_batch{batch_threshold:.2f}"
                    run_experiment(config, exp_name)
                
                # Vary confidence_threshold
                for confidence_threshold in confidence_thresholds:
                    config = base_config.copy()
                    config["dataset"] = dataset
                    config["algorithm"] = algorithm
                    config["confidence_threshold"] = confidence_threshold
                    config["batch_threshold"] = 0.9  # Default
                    config["output_dir"] = os.path.join(base_output_dir, 
                                                       f"{dataset}_{algorithm}_conf{confidence_threshold:.2f}")
                    
                    exp_name = f"{dataset}_{algorithm}_conf{confidence_threshold:.2f}"
                    run_experiment(config, exp_name)

    print("\n\nAll experiments completed!")
    print(f"Results stored in directory: {base_output_dir}")

if __name__ == "__main__":
    main()