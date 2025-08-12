#!/usr/bin/env python3
"""
Plot iteration-frame graphs from SPH solver convergence logs.

This script reads log/iterations_type_*.json files and creates visualization
plots showing convergence behavior for different matrix types.
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_log_data(log_dir="log"):
    """Load all iteration log files from the log directory."""
    data = {}
    
    # Find all iterations_type_*.json files
    pattern = os.path.join(log_dir, "iterations_type_*.json")
    log_files = glob.glob(pattern)
    
    if not log_files:
        print(f"No log files found in {log_dir}/")
        return data
    
    for log_file in sorted(log_files):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                matrix_type = log_data["matrix_type"]
                
                # Extract frame and iteration data
                frames = [entry[0] for entry in log_data["data"]]
                iterations = [entry[1] for entry in log_data["data"]]
                
                data[matrix_type] = {
                    "frames": frames,
                    "iterations": iterations,
                    "avg_iterations": log_data["avg_iterations"],
                    "total_frames": log_data["total_frames"]
                }
                
                print(f"Loaded {len(frames)} data points for matrix type {matrix_type}")
                
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return data

def create_plots(data, output_dir="plots"):
    """Create iteration-frame plots for each matrix type."""
    if not data:
        print("No data to plot")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Define colors and labels for different matrix types
    type_info = {
        0: {"color": "blue", "label": "Type 0 (B)"},
        1: {"color": "red", "label": "Type 1 (√D B √D)"},
        2: {"color": "green", "label": "Type 2 (D B D)"},
        3: {"color": "orange", "label": "Type 3 (A non-sym)"},
        "ADMM": {"color": "purple", "label": "ADMM"},
        "NormalEq": {"color": "brown", "label": "Normal Equations"},
        "Barrier": {"color": "pink", "label": "Barrier"}
    }
    
    # 1. Individual plots for each matrix type
    for matrix_type, type_data in data.items():
        plt.figure(figsize=(12, 6))
        
        info = type_info.get(matrix_type, {"color": "gray", "label": f"Type {matrix_type}"})
        
        plt.plot(type_data["frames"], type_data["iterations"], 
                color=info["color"], linewidth=1.5, alpha=0.8, label=info["label"])
        
        # Add average line
        avg_iter = type_data["avg_iterations"]
        plt.axhline(y=avg_iter, color=info["color"], linestyle='--', alpha=0.8, 
                   label=f"Average: {avg_iter:.2f}")
        
        plt.xlabel("Frame")
        plt.ylabel("Iterations to Convergence")
        plt.title(f"Convergence Analysis - {info['label']}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics text
        stats_text = f"Total Frames: {type_data['total_frames']}\n"
        stats_text += f"Avg Iterations: {avg_iter:.2f}\n"
        stats_text += f"Min Iterations: {min(type_data['iterations'])}\n"
        stats_text += f"Max Iterations: {max(type_data['iterations'])}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save individual plot
        filename = f"iterations_type_{matrix_type}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        plt.close()
    
    # 2. Combined comparison plot
    if len(data) > 1:
        plt.figure(figsize=(15, 8))
        
        for matrix_type, type_data in data.items():
            info = type_info.get(matrix_type, {"color": "gray", "label": f"Type {matrix_type}"})
            
            plt.plot(type_data["frames"], type_data["iterations"], 
                    color=info["color"], linewidth=1.5, alpha=0.8, 
                    label=f"{info['label']} (avg: {type_data['avg_iterations']:.2f})")
        
        plt.xlabel("Frame")
        plt.ylabel("Iterations to Convergence")
        plt.title("Convergence Comparison - All Matrix Types")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_filepath = os.path.join(output_dir, "iterations_comparison.png")
        plt.savefig(comparison_filepath, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {comparison_filepath}")
        plt.close()
    
    # 3. Statistics summary plot (bar chart)
    if len(data) > 1:
        matrix_types = list(data.keys())
        avg_iterations = [data[mt]["avg_iterations"] for mt in matrix_types]
        colors = [type_info.get(mt, {"color": "gray"})["color"] for mt in matrix_types]
        labels = [type_info.get(mt, {"label": f"Type {mt}"})["label"] for mt in matrix_types]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(matrix_types)), avg_iterations, color=colors, alpha=0.7)
        
        plt.xlabel("Matrix Type")
        plt.ylabel("Average Iterations")
        plt.title("Average Convergence Performance Comparison")
        plt.xticks(range(len(matrix_types)), labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, avg) in enumerate(zip(bars, avg_iterations)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{avg:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save statistics plot
        stats_filepath = os.path.join(output_dir, "iterations_statistics.png")
        plt.savefig(stats_filepath, dpi=300, bbox_inches='tight')
        print(f"Saved statistics plot: {stats_filepath}")
        plt.close()

def print_summary(data):
    """Print a summary of the loaded data."""
    if not data:
        print("No data loaded")
        return
    
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    
    for matrix_type, type_data in data.items():
        print(f"\nMatrix Type {matrix_type}:")
        print(f"  Total frames: {type_data['total_frames']}")
        print(f"  Average iterations: {type_data['avg_iterations']:.3f}")
        print(f"  Min iterations: {min(type_data['iterations'])}")
        print(f"  Max iterations: {max(type_data['iterations'])}")
        print(f"  Std deviation: {np.std(type_data['iterations']):.3f}")

def main():
    """Main function to run the plotting script."""
    print("SPH Solver Convergence Analysis")
    print("-" * 40)
    
    # Check if log directory exists
    log_dir = "log"
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' not found!")
        print("Make sure you have run the simulation with logging enabled.")
        return
    
    # Load data
    print(f"Loading data from {log_dir}/...")
    data = load_log_data(log_dir)
    
    if not data:
        print("No valid log files found!")
        return
    
    # Print summary
    print_summary(data)
    
    # Create plots
    print(f"\nCreating plots...")
    create_plots(data)
    
    print(f"\nAnalysis complete! Check the 'plots/' directory for generated graphs.")

if __name__ == "__main__":
    main()