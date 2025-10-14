# SPH Taichi (In Progress)

A high-performance Smoothed Particle Hydrodynamics (SPH) simulator built with [Taichi](https://github.com/taichi-dev/taichi).

This project is currently under active development. This document serves as a guide for developers contributing to the project.

## Implemented Features

Our simulator has a robust set of features, divided into the core simulation engine and the specific physics models implemented.

### Core Engine
*   **Cross-Platform Support**: Runs on both Windows and Linux.
*   **GPU Acceleration**: Leverages Taichi to perform massively parallel computations on the GPU.
*   **Optimized Neighborhood Search**: Implements a fast neighborhood search using a GPU-accelerated parallel prefix sum and counting sort algorithm.
*   **Particle Emitter System**: Supports configurable box and cylinder-shaped emitters, featuring an optional particle recycling system that re-emits particles that have left a defined bounding box, allowing for continuous emission in long-running simulations.
*   **Caching System**: Provides an in-memory, frame-level caching system that captures snapshots of the  simulation state (particles, emitters, solver time, etc.).

### SPH Simulation Features
*   **Viscosity**: Implements a viscosity model [1].
*   **Surface Tension**: Implements a surface tension model [2].
*   **Fluid-Rigid Coupling**: Supports both one-way and two-way coupling between fluids and rigid bodies.
*   **Boundary Conditions**: Handles boundary conditions based on the methods described in [3].

---
```
python main.py --scene_file ./data/scenes/tests/fluid_cube.json
```

## Development Roadmap

The following is a list of tasks and areas for improvement that we need to address.

### Planned Revisions & Enhancements
*   **Animation System**: The current animation-related code is scattered across the project and lacks a unified structure. It needs to be refactored into a single, cohesive module.
*   **Divergence-Free Condition**: The divergence-free condition for the DFSPH solver has been implemented but is not yet fully utilized. It needs to be properly integrated to be usable in simulations.

### Code Refactoring
*   **Modular Architecture**: Refactor the codebase to improve modularity. This includes:
    *   Separating the unified SPH solver from an operator-splitting-based approach.
    *   Clearly defining responsibilities for simulation solvers (`IISPH.py`, `DFSPH.py`, etc.).
    *   Organizing utility modules (e.g., emitter, plotting tools).
    *   Standardizing data structures for particles (e.g., position, velocity, pressure).
*   **Utility System Refinement**: Alongside the Animation System refactor, other utilities should be reviewed for cleanup and better integration.

### Project Management & Organization
*   **Scene and Model Organization**: The `data/scenes` and `data/models` directories need to be organized for clarity and ease of use.
*   **Asset Management**: Develop a more systematic way to manage scenes, models, and other assets.

---
#### *References:*
1. SCHECHTER H., BRIDSON R.: Ghost sph for animating water. ACM Transactions on Graphics (TOG) 31, 4 (2012), 
1–8. 5
2. AKINCI N., CORNELIS J., AKINCI G., TESCHNER M.: Coupling elastic solids with smoothed particle hydrodynamics 
fluids. Computer Animation and Virtual Worlds 24, 3-4 (2013), 195–203. 5, 8
3. BENDER J., WESTHOFEN L., JESKE S. R.: Consistent sph rigid-fluid coupling. In VMV (2023), pp. 209–217. 2, 5
 