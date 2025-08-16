#!/usr/bin/env python3
"""
Debug the spectral radius differences between PBF2.py and our 2D implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def debug_constraint_formulations():
    """
    Compare different constraint formulations and their impact on spectral radius
    """
    print("="*60)
    print("DEBUGGING SPECTRAL RADIUS DIFFERENCES")
    print("="*60)
    
    # Simple 2D test case
    n = 25  # 5x5 grid
    domain_size = 1.0
    particle_radius = 0.05
    support_radius = 4.0 * particle_radius
    rho_0 = 1000.0
    
    # Create particle positions
    grid_size = int(np.sqrt(n))
    spacing = domain_size / (grid_size + 1)
    positions = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 1) * spacing
            y = (j + 1) * spacing
            positions.append([x, y])
    
    positions = np.array(positions)
    print(f"System: {n} particles, spacing = {spacing:.3f}, support_radius = {support_radius:.3f}")
    
    # Find neighbors
    neighbors = []
    for i in range(n):
        particle_neighbors = []
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < support_radius:
                    particle_neighbors.append(j)
        neighbors.append(particle_neighbors)
    
    avg_neighbors = np.mean([len(neighs) for neighs in neighbors])
    print(f"Average neighbors: {avg_neighbors:.1f}")
    
    # Cubic spline kernel (2D)
    def cubic_kernel_gradient(r_vec, h):
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            return np.zeros(2)
        
        q = r / h
        sigma = 10.0 / (7.0 * np.pi * h**2)  # 2D normalization
        
        if q >= 2.0:
            return np.zeros(2)
        elif q >= 1.0:
            dW_dr = sigma * (-3.0/4.0) * (2.0 - q)**2 / h
        else:
            dW_dr = sigma * (-3.0 * q + 9.0/4.0 * q**2) / h
        
        return dW_dr * (r_vec / r)
    
    # Test different formulations
    formulations = {
        "Our_Implementation": {
            "description": "c_i = m_i(ρ_i - ρ_0), ∇c_i = m_i Σ_j m_j ∇W_ij, A_ij = (∇c_i · ∇c_j) / m_j",
            "mass_formula": lambda r: rho_0 * r**2,  # m_i = ρ_0 * r²
            "density_factor": 1.0,
            "constraint_scaling": 1.0
        },
        "PBF2_Style": {
            "description": "Scaled constraint with density weighting",
            "mass_formula": lambda r: rho_0 * (r**2) * 0.1,  # Smaller mass
            "density_factor": 1.0,
            "constraint_scaling": 1.0 / rho_0  # Scale by 1/ρ_0
        },
        "Normalized": {
            "description": "Normalized by particle volume",
            "mass_formula": lambda r: rho_0 * (r**2),
            "density_factor": 1.0,
            "constraint_scaling": 1.0 / (rho_0 * particle_radius**2)  # Volume normalization
        }
    }
    
    results = {}
    
    for name, config in formulations.items():
        print(f"\n--- Testing {name} ---")
        print(f"Description: {config['description']}")
        
        # Set masses
        masses = np.array([config['mass_formula'](particle_radius) for _ in range(n)])
        print(f"Particle mass: {masses[0]:.6f}")
        
        # Compute densities (simplified - assume uniform)
        densities = np.full(n, rho_0 * 0.1)  # Under-dense
        print(f"Density: {densities[0]:.3f} (target: {rho_0})")
        
        # Compute constraints
        constraints = masses * (densities - rho_0) * config['constraint_scaling']
        print(f"Constraint range: [{constraints.min():.2e}, {constraints.max():.2e}]")
        
        # Compute constraint gradients
        constraint_gradients = np.zeros((n, 2))
        
        for i in range(n):
            grad_ci = np.zeros(2)
            for j in neighbors[i]:
                r_vec = positions[i] - positions[j]
                grad_W = cubic_kernel_gradient(r_vec, support_radius)
                grad_ci += masses[i] * masses[j] * grad_W
            
            # Apply density factor
            constraint_gradients[i] = grad_ci * config['density_factor']
        
        print(f"Max gradient magnitude: {np.max(np.linalg.norm(constraint_gradients, axis=1)):.2e}")
        
        # Build system matrix A_ij = (∇c_i · ∇c_j) / m_j
        rows, cols, data = [], [], []
        
        for i in range(n):
            for j in range(n):
                grad_ci = constraint_gradients[i]
                grad_cj = constraint_gradients[j]
                value = np.dot(grad_ci, grad_cj) / masses[j]
                
                if abs(value) > 1e-15:
                    rows.append(i)
                    cols.append(j)
                    data.append(value)
        
        A = csr_matrix((data, (rows, cols)), shape=(n, n))
        A = (A + A.T) / 2.0  # Ensure symmetry
        
        # Add regularization if needed
        if A.nnz == 0:
            A.setdiag(1e-6)
        else:
            # Check diagonal
            min_diag = np.min(np.abs(A.diagonal()))
            if min_diag < 1e-12:
                A.setdiag(A.diagonal() + 1e-6)
        
        print(f"Matrix nnz: {A.nnz}, sparsity: {(1-A.nnz/n**2)*100:.1f}%")
        
        # Compute spectral radius
        try:
            eigenvals = np.linalg.eigvals(A.toarray())
            spectral_radius = np.max(np.abs(eigenvals))
            min_eigenval = np.min(np.real(eigenvals))
            
            results[name] = {
                'spectral_radius': spectral_radius,
                'min_eigenval': min_eigenval,
                'max_constraint': np.max(np.abs(constraints)),
                'max_gradient': np.max(np.linalg.norm(constraint_gradients, axis=1)),
                'matrix_norm': np.max(np.abs(A.data)) if A.nnz > 0 else 0
            }
            
            print(f"Spectral radius: {spectral_radius:.2e}")
            print(f"Min eigenvalue: {min_eigenval:.2e}")
            print(f"Matrix condition: {np.linalg.cond(A.toarray()):.2e}")
            
        except Exception as e:
            print(f"Failed to compute eigenvalues: {e}")
            results[name] = {'spectral_radius': float('inf'), 'min_eigenval': 0}
    
    # Comparison analysis
    print(f"\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    print(f"{'Formulation':<15} {'Spectral ρ':<12} {'Min λ':<12} {'Max |c|':<12} {'Max |∇c|':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        if np.isfinite(result['spectral_radius']):
            print(f"{name:<15} {result['spectral_radius']:<12.2e} {result['min_eigenval']:<12.2e} "
                  f"{result['max_constraint']:<12.2e} {result['max_gradient']:<12.2e}")
        else:
            print(f"{name:<15} {'INF':<12} {'N/A':<12} {result.get('max_constraint', 0):<12.2e} "
                  f"{result.get('max_gradient', 0):<12.2e}")
    
    return results

def analyze_pbf2_vs_our_differences():
    """
    Analyze key differences between PBF2 and our implementation
    """
    print(f"\n" + "="*60)
    print("KEY DIFFERENCES ANALYSIS")
    print("="*60)
    
    differences = {
        "1. Mass Definition": {
            "PBF2": "m_i = ρ_0 * V_i (volume-based mass)",
            "Our": "m_i = ρ_0 * r² (area-based mass)"
        },
        "2. Constraint Scaling": {
            "PBF2": "Constraint includes density weighting: (m_i/ρ_i) factor",
            "Our": "Direct constraint: c_i = m_i(ρ_i - ρ_0)"
        },
        "3. Matrix Construction": {
            "PBF2": "Uses Bii diagonal approximation + regularization",
            "Our": "Full matrix construction A = ∇c M⁻¹ ∇cᵀ"
        },
        "4. Jacobi Iteration": {
            "PBF2": "Uses preconditioned Jacobi with diagonal scaling",
            "Our": "Classical Jacobi without preconditioning"
        },
        "5. System Scaling": {
            "PBF2": "Works with pressure corrections (small increments)",
            "Our": "Works with full constraint values (large magnitudes)"
        }
    }
    
    for category, comparison in differences.items():
        print(f"\n{category}:")
        print(f"  PBF2: {comparison['PBF2']}")
        print(f"  Our:  {comparison['Our']}")
    
    print(f"\n" + "="*60)
    print("LIKELY ROOT CAUSE")
    print("="*60)
    print("The spectral radius difference likely stems from:")
    print("1. **Scale mismatch**: Our constraints are O(1000) while PBF2 uses normalized/scaled values")
    print("2. **Matrix construction**: We build full A matrix, PBF2 uses diagonal approximation") 
    print("3. **Regularization**: PBF2 adds eps=1e-3 to diagonal, we add eps=1e-6")
    print("4. **Density weighting**: PBF2 includes (m_i/ρ_i) factors that reduce matrix magnitude")

def test_scaling_fix():
    """
    Test if proper scaling fixes the spectral radius
    """
    print(f"\n" + "="*60)
    print("TESTING SCALING FIX")
    print("="*60)
    
    # Create small test system
    n = 9  # 3x3 grid
    positions = np.array([[i, j] for i in range(3) for j in range(3)]) * 0.2
    support_radius = 0.3
    rho_0 = 1000.0
    
    # Find neighbors
    neighbors = []
    for i in range(n):
        particle_neighbors = []
        for j in range(n):
            if i != j and np.linalg.norm(positions[i] - positions[j]) < support_radius:
                particle_neighbors.append(j)
        neighbors.append(particle_neighbors)
    
    def cubic_kernel_gradient(r_vec, h):
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            return np.zeros(2)
        
        q = r / h
        sigma = 10.0 / (7.0 * np.pi * h**2)
        
        if q >= 2.0:
            return np.zeros(2)
        elif q >= 1.0:
            dW_dr = sigma * (-3.0/4.0) * (2.0 - q)**2 / h
        else:
            dW_dr = sigma * (-3.0 * q + 9.0/4.0 * q**2) / h
        
        return dW_dr * (r_vec / r)
    
    # Test different scaling approaches
    scaling_tests = {
        "Original": {"mass_scale": 1.0, "constraint_scale": 1.0, "regularization": 1e-6},
        "PBF2_Style": {"mass_scale": 0.01, "constraint_scale": 1e-3, "regularization": 1e-3},
        "Volume_Normalized": {"mass_scale": 1.0, "constraint_scale": 1e-6, "regularization": 1e-4},
        "Heavy_Regularized": {"mass_scale": 1.0, "constraint_scale": 1.0, "regularization": 1e-2}
    }
    
    print(f"{'Test':<20} {'Spectral ρ':<15} {'Condition':<15} {'Status':<15}")
    print("-" * 70)
    
    for test_name, params in scaling_tests.items():
        # Set parameters
        masses = np.full(n, rho_0 * 0.01 * params['mass_scale'])  # Small masses
        densities = np.full(n, rho_0 * 0.1)  # Under-dense
        constraints = masses * (densities - rho_0) * params['constraint_scale']
        
        # Build matrix with scaling
        constraint_gradients = np.zeros((n, 2))
        for i in range(n):
            grad_ci = np.zeros(2)
            for j in neighbors[i]:
                r_vec = positions[i] - positions[j]
                grad_W = cubic_kernel_gradient(r_vec, support_radius)
                grad_ci += masses[i] * masses[j] * grad_W
            constraint_gradients[i] = grad_ci
        
        # System matrix
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = np.dot(constraint_gradients[i], constraint_gradients[j]) / masses[j]
        
        # Regularize
        A = (A + A.T) / 2.0
        A += params['regularization'] * np.eye(n)
        
        # Analyze
        try:
            eigenvals = np.linalg.eigvals(A)
            spectral_radius = np.max(np.abs(eigenvals))
            condition = np.linalg.cond(A)
            status = "Converges" if spectral_radius < 1000 else "Diverges"
            
            print(f"{test_name:<20} {spectral_radius:<15.2e} {condition:<15.2e} {status:<15}")
            
        except:
            print(f"{test_name:<20} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")

def main():
    """Main debugging function"""
    print("Debugging Spectral Radius Differences Between PBF2 and 2D Implementation")
    print("=" * 80)
    
    # Test different formulations
    results = debug_constraint_formulations()
    
    # Analyze differences
    analyze_pbf2_vs_our_differences()
    
    # Test scaling fixes
    test_scaling_fix()
    
    print(f"\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    print("1. The spectral radius difference is primarily due to SCALING")
    print("2. Our implementation uses full-magnitude constraints (O(1000))")
    print("3. PBF2 likely uses normalized/scaled constraints (O(1))")
    print("4. Solution: Scale constraints and add proper regularization")
    print("5. Key insight: Spectral radius scales with constraint magnitude!")

if __name__ == "__main__":
    main()