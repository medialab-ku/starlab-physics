#!/usr/bin/env python3
"""
Test script for the Normal Equations approach in PBF2.
This demonstrates how to use the new method=3 option.
"""

import taichi as ti
from PBF2 import PBF2Solver
from particle_system import ParticleSystem
import json

def test_normal_equations():
    """Test the Normal Equations approach"""
    
    # Initialize Taichi
    ti.init(arch=ti.cpu)
    
    # Load a scene configuration
    scene_file = "data/scenes/dragon_bath.json"
    
    try:
        with open(scene_file, 'r') as f:
            scene_config = json.load(f)
    except FileNotFoundError:
        print(f"Scene file {scene_file} not found. Using default configuration.")
        scene_config = {
            "particleRadius": 0.025,
            "density0": 1000.0,
            "timeStepSize": 0.002,
            "numSubsteps": 5,
            "numIters": 20
        }
    
    # Create particle system
    ps = ParticleSystem(scene_config)
    
    # Create PBF2 solver
    solver = PBF2Solver(ps)
    
    # Set method to Normal Equations (method 3)
    solver.method = 3
    print(f"Using method {solver.method}: Normal Equations")
    
    # Run a few simulation steps
    print("Running simulation with Normal Equations approach...")
    
    for step in range(5):
        print(f"Step {step + 1}")
        solver.substep()
        
        # Print some statistics
        if hasattr(solver, 'stats_iter'):
            print(f"  Iterations: {solver.stats_iter}")
    
    print("Normal Equations test completed!")
    
    # Clean up
    ti.reset()

if __name__ == "__main__":
    test_normal_equations()
