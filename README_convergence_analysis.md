# SPH Solver Convergence Analysis

This directory contains tools for analyzing the convergence behavior of different matrix formulations in the SPH solver.

## Usage Instructions

### 1. Collect Convergence Data

1. **Start the simulation**:
   ```bash
   python run_simulation.py --scene_file ./data/scenes/multi_fluid.json
   ```

2. **Enable logging**: Check the "Enable logging" checkbox in the GUI

3. **Test different matrix types**: 
   - Use the "mat type" slider (0-3) when "Projected Jacobi" is selected
   - Try different solver methods (0=ProjectedJacobi, 1=ADMM, 2=Barrier)

4. **Save data**: Press 'r' key or "Reset & Save logs" button to save current data

### 2. Analyze Results

Run the analysis script:
```bash
python plot_iterations.py
```

This will:
- Load all `log/iterations_type_*.json` files
- Generate individual plots for each matrix type
- Create comparison plots if multiple types were tested
- Print convergence statistics

### 3. Output Files

**Log Files** (in `log/` directory):
- `iterations_type_0.json` - Matrix type 0 (B)
- `iterations_type_1.json` - Matrix type 1 (√D B √D)
- `iterations_type_2.json` - Matrix type 2 (D B D)  
- `iterations_type_3.json` - Matrix type 3 (A non-symmetric)
- `iterations_type_ADMM.json` - ADMM method
- `iterations_combined.json` - All data combined

**Plot Files** (in `plots/` directory):
- `iterations_type_*.png` - Individual convergence plots
- `iterations_comparison.png` - All methods compared
- `iterations_statistics.png` - Average performance bar chart

### 4. Matrix Types Explained

- **Type 0 (B)**: `J M⁻¹ Jᵀ y = b` - Original symmetric formulation
- **Type 1 (√D B √D)**: `√D J M⁻¹ Jᵀ √D x₁ = √D b` - Square root scaling
- **Type 2 (D B D)**: `D J M⁻¹ Jᵀ D x₂ = D b` - Full density scaling
- **Type 3 (A non-symmetric)**: `J M⁻¹ Jᵀ D p = b` - Non-symmetric baseline

Where `D_ii = m_i / ρ_i²` is the density-weighting matrix.

### 5. Example Analysis Workflow

```bash
# 1. Run simulation, enable logging, test different matrix types
python run_simulation.py --scene_file ./data/scenes/multi_fluid.json

# 2. Generate analysis plots
python plot_iterations.py

# 3. View results
ls plots/
# iterations_type_0.png, iterations_comparison.png, etc.
```

### 6. Interpreting Results

- **Lower iteration counts** = faster convergence = better numerical performance
- **Consistent iteration counts** = stable convergence behavior
- **High variance** = unstable or poorly conditioned system
- **Comparison plots** help identify which matrix formulation performs best

The analysis helps validate that different matrix formulations solve the same physical system while providing different numerical characteristics.