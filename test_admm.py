import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

np.random.seed(42)

n_particles = 20
radius = 0.1

positions = np.random.uniform(0, 2, (n_particles, 2))
v0_particles = np.random.randn(n_particles, 2) * 0.5

print(f"Generated {n_particles} particles in (0,0)~(2,2) grid")
print(f"Particle radius: {radius}")
print("Particle positions:")
for i, pos in enumerate(positions):
    print(f"  Particle {i}: {pos}")

def detect_collisions(positions, radius):
    collisions = []
    collision_pairs = set()
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < 2 * radius:
                displacement = positions[i] - positions[j]
                if distance > 1e-10:
                    normal = displacement / distance
                else:
                    normal = np.array([1.0, 0.0])
                
                collisions.append({
                    'i': i, 'j': j, 
                    'distance': distance,
                    'overlap': 2 * radius - distance,
                    'normal': normal,
                    'displacement': displacement
                })
                collision_pairs.add(i)
                collision_pairs.add(j)
    
    return collisions, collision_pairs

collisions, collision_pairs = detect_collisions(positions, radius)

print(f"\nCollision Detection:")
print(f"Number of collisions: {len(collisions)}")
print(f"Colliding particles: {sorted(collision_pairs)}")

for k, collision in enumerate(collisions):
    i, j = collision['i'], collision['j']
    dist = collision['distance']
    overlap = collision['overlap']
    normal = collision['normal']
    print(f"  Collision {k}: particles {i}-{j}, distance={dist:.3f}, overlap={overlap:.3f}")
    print(f"    Normal: {normal}")

def construct_constraint_matrix(collisions, n_particles, duplicate_fraction=0.3):
    """
    Construct constraint matrix J with some duplicated constraints to test rank deficiency
    
    Args:
        collisions: List of collision data
        n_particles: Number of particles
        duplicate_fraction: Fraction of constraints to duplicate (0.0 to 1.0)
    """
    n_original = len(collisions)
    n_duplicates = int(n_original * duplicate_fraction)
    n_total_constraints = n_original + n_duplicates
    n_vars = 2 * n_particles
    
    J = np.zeros((n_total_constraints, n_vars))
    
    # Add original constraints
    for k, collision in enumerate(collisions):
        i, j = collision['i'], collision['j']
        normal = collision['normal']
        
        J[k, 2*i:2*i+2] = normal
        J[k, 2*j:2*j+2] = -normal
    
    # Add duplicated constraints
    duplicate_indices = np.random.choice(n_original, n_duplicates, replace=True)
    print(f"\nDuplicating constraints: {duplicate_indices}")
    
    for k, orig_idx in enumerate(duplicate_indices):
        dup_row = n_original + k
        collision = collisions[orig_idx]
        i, j = collision['i'], collision['j']
        normal = collision['normal']
        
        # Add exact duplicate of constraint
        J[dup_row, 2*i:2*i+2] = normal
        J[dup_row, 2*j:2*j+2] = -normal
        
        print(f"  Duplicated constraint {orig_idx} (particles {i}-{j}) as row {dup_row}")
    
    return J, n_duplicates

if len(collisions) > 0:
    J, n_duplicates = construct_constraint_matrix(collisions, n_particles, duplicate_fraction=0.4)
    
    print(f"\nConstraint matrix J (with duplicates):")
    print(f"Shape: {J.shape}")
    print(f"Original constraints: {len(collisions)}")
    print(f"Duplicated constraints: {n_duplicates}")
    print(f"Total constraints: {J.shape[0]}")
    print(f"Rank of J: {np.linalg.matrix_rank(J)}")
    print(f"Expected rank deficiency: {n_duplicates}")
    print("J matrix (first 5 rows, first 10 cols):")
    print(J[:min(5, J.shape[0]), :min(10, J.shape[1])])
    
    JJT = J @ J.T
    print(f"\nAnalyzing J @ J.T (with duplicated constraints):")
    print(f"Shape: {JJT.shape}")
    print(f"Rank of J @ J.T: {np.linalg.matrix_rank(JJT)}")
    print(f"Determinant of J @ J.T: {np.linalg.det(JJT)}")
    eigenvals = np.linalg.eigvals(JJT)
    print(f"Number of near-zero eigenvalues: {np.sum(eigenvals < 1e-10)}")
    print(f"Smallest eigenvalues: {np.sort(eigenvals)[:5]}")
    print(f"Largest eigenvalues: {np.sort(eigenvals)[-5:]}")
    print(f"Is J @ J.T singular? {np.linalg.det(JJT) < 1e-10}")
    
    # Detailed comparison: J*J^T vs I + J^T*J
    rho = 1.0
    I_small = np.eye(J.shape[0])  # For J*J^T
    I_large = np.eye(J.shape[1])  # For J^T*J
    
    JTJ = J.T @ J
    A = I_large + rho * JTJ
    
    print(f"\n" + "="*80)
    print("DETAILED MATRIX COMPARISON: J*J^T vs I + J^T*J")
    print("="*80)
    
    print(f"\n1. MATRIX DIMENSIONS:")
    print(f"   J shape: {J.shape}")
    print(f"   J*J^T shape: {JJT.shape} (constraint space)")
    print(f"   J^T*J shape: {JTJ.shape} (velocity space)")
    print(f"   I + J^T*J shape: {A.shape}")
    
    print(f"\n2. RANK ANALYSIS:")
    print(f"   Rank of J: {np.linalg.matrix_rank(J)}")
    print(f"   Rank of J*J^T: {np.linalg.matrix_rank(JJT)}")
    print(f"   Rank of J^T*J: {np.linalg.matrix_rank(JTJ)}")
    print(f"   Rank of I + J^T*J: {np.linalg.matrix_rank(A)}")
    
    print(f"\n3. SINGULARITY TEST:")
    print(f"   det(J*J^T) = {np.linalg.det(JJT):.2e}")
    print(f"   det(J^T*J) = {np.linalg.det(JTJ):.2e}")
    print(f"   det(I + J^T*J) = {np.linalg.det(A):.2e}")
    print(f"   Is J*J^T singular? {np.linalg.det(JJT) < 1e-10}")
    print(f"   Is J^T*J singular? {np.linalg.det(JTJ) < 1e-10}")
    print(f"   Is I + J^T*J singular? {np.linalg.det(A) < 1e-10}")
    
    print(f"\n4. EIGENVALUE ANALYSIS:")
    eigs_JJT = np.linalg.eigvals(JJT)
    eigs_JTJ = np.linalg.eigvals(JTJ)
    eigs_A = np.linalg.eigvals(A)
    
    print(f"   J*J^T eigenvalues (sorted):")
    eigs_JJT_sorted = np.sort(eigs_JJT)
    print(f"     Smallest 5: {eigs_JJT_sorted[:5]}")
    print(f"     Largest 5:  {eigs_JJT_sorted[-5:]}")
    print(f"     Near-zero (<1e-10): {np.sum(eigs_JJT < 1e-10)}")
    
    print(f"   J^T*J eigenvalues (sorted):")
    eigs_JTJ_sorted = np.sort(eigs_JTJ)
    print(f"     Smallest 5: {eigs_JTJ_sorted[:5]}")
    print(f"     Largest 5:  {eigs_JTJ_sorted[-5:]}")
    print(f"     Near-zero (<1e-10): {np.sum(eigs_JTJ < 1e-10)}")
    
    print(f"   I + J^T*J eigenvalues (sorted):")
    eigs_A_sorted = np.sort(eigs_A)
    print(f"     Smallest 5: {eigs_A_sorted[:5]}")
    print(f"     Largest 5:  {eigs_A_sorted[-5:]}")
    print(f"     Near-zero (<1e-10): {np.sum(eigs_A < 1e-10)}")
    
    print(f"\n5. CONDITION NUMBERS:")
    print(f"   cond(J*J^T) = {np.linalg.cond(JJT):.2e}")
    print(f"   cond(J^T*J) = {np.linalg.cond(JTJ):.2e}")
    print(f"   cond(I + J^T*J) = {np.linalg.cond(A):.2e}")
    
    print(f"\n6. KEY INSIGHT:")
    print(f"   • J*J^T is singular (has {np.sum(eigs_JJT < 1e-10)} zero eigenvalues)")
    print(f"   • J^T*J is also singular (has {np.sum(eigs_JTJ < 1e-10)} zero eigenvalues)")
    print(f"   • But I + J^T*J is NON-SINGULAR!")
    print(f"   • The identity matrix I shifts all eigenvalues by +1")
    print(f"   • This makes ADMM v-update well-defined even with rank deficient J")
    
    # Show the relationship between eigenvalues
    print(f"\n7. EIGENVALUE RELATIONSHIP:")
    print(f"   J and J^T have same non-zero eigenvalues (different multiplicities)")
    print(f"   If λ is eigenvalue of J^T*J, then λ+1 is eigenvalue of I + J^T*J")
    print(f"   Verification:")
    nonzero_JTJ = eigs_JTJ[eigs_JTJ > 1e-10]
    shifted_eigs = nonzero_JTJ + 1
    corresponding_A = eigs_A[eigs_A > 1.1]  # Those that came from J^T*J + I
    print(f"     Non-zero eigs of J^T*J: {np.sort(nonzero_JTJ)}")
    print(f"     Shifted by +1:         {np.sort(shifted_eigs)}")
    print(f"     Corresponding in A:    {np.sort(corresponding_A)}")
    
else:
    print(f"\nNo collisions detected! All particles are separated by > {2*radius}")
    J = np.zeros((0, 2*n_particles))
    n_duplicates = 0

def admm_solver_separation(v0, J, rho=1.0, max_iter=1000, tol=1e-6):
    """
    ADMM solver for: min_v (1/2)||v - v0||^2 subject to Jv >= 0
    
    Problem formulation:
    - Primal variable: v (velocity)
    - Constraint: Jv >= 0 (separation constraint)
    - Objective: minimize deviation from initial velocity v0
    
    ADMM reformulation: min_v (1/2)||v - v0||^2 + I_+(Jv)
    where I_+(z) = 0 if z >= 0, +∞ otherwise
    
    Augmented Lagrangian: L = (1/2)||v - v0||^2 + u^T(Jv - z) + (rho/2)||Jv - z||^2
    """
    n_vars = len(v0)
    n_constraints = J.shape[0]
    
    # Initialize ADMM variables
    v = v0.copy()                    # Primal variable (velocities)
    z = np.zeros(n_constraints)      # Auxiliary variable for constraint Jv >= 0
    u = np.zeros(n_constraints)      # Dual variable (Lagrange multipliers)
    
    # Pre-compute matrix inverse for v-update (stays constant throughout iterations)
    # The v-update requires solving: (I + rho*J^T*J)v = v0 + rho*J^T*(z - u)
    I = np.eye(n_vars)
    JTJ_inv = np.linalg.inv(I + rho * J.T @ J)
    
    print(f"\nStarting ADMM for separation (Jv >= 0) with rho={rho}")
    
    for k in range(max_iter):
        v_old = v.copy()  # Store previous v for dual residual computation
        
        # STEP 1: v-update (minimize over v with z, u fixed)
        # Solve: min_v (1/2)||v - v0||^2 + u^T*Jv + (rho/2)||Jv - z||^2
        # Taking derivative w.r.t. v and setting to 0:
        # (v - v0) + J^T*u + rho*J^T*(Jv - z) = 0
        # (I + rho*J^T*J)v = v0 + rho*J^T*(z - u)
        v = JTJ_inv @ (v0 + rho * J.T @ (z - u))
        
        # STEP 2: z-update (minimize over z with v, u fixed) 
        # Solve: min_z u^T*(Jv - z) + (rho/2)||Jv - z||^2 + I_+(z)
        # This separates per constraint: min_z_i u_i*(Jv_i - z_i) + (rho/2)(Jv_i - z_i)^2 + I_+(z_i)
        # Optimal z_i = max(0, Jv_i + u_i/rho)
        Jv_u = J @ v + u / rho  # Note: we use u/rho instead of just u in the formula
        z = np.maximum(0, Jv_u)  # Projection onto non-negative orthant for >= 0 constraint
        
        # STEP 3: u-update (dual variable update)
        # Standard ADMM dual update: u := u + rho*(Jv - z)
        # This enforces the constraint Jv = z in the limit
        u = u + rho * (J @ v - z)
        
        # CONVERGENCE CHECK: Compute primal and dual residuals
        # Primal residual: ||Jv - z|| (measures constraint violation)
        primal_residual = np.linalg.norm(J @ v - z)
        
        # Dual residual: ||rho * J^T * (z_new - z_old)|| (measures change in dual)
        # Since z_old is implicit, we compute it as the difference in dual updates
        dual_residual = np.linalg.norm(rho * J.T @ (z - (J @ v_old + u / rho - (J @ v_old))))
        
        # if k % 100 == 0 or k < 10:
        #     print(f"Iter {k}: primal_res={primal_residual:.6f}, dual_res={dual_residual:.6f}")
        #     print(f"  v: {v}")
        #     print(f"  Jv: {J @ v} (should be >= 0)")
        
        # Check convergence: both residuals must be small
        if primal_residual < tol and dual_residual < tol:
            print(f"Converged at iteration {k}")
            break
    
    return v, z, u


def visualize_particles(positions, radius, collision_pairs, collisions):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    for i, pos in enumerate(positions):
        if i in collision_pairs:
            color = 'red'
            alpha = 0.7
        else:
            color = 'blue'
            alpha = 0.5
        
        circle = Circle(pos, radius, color=color, alpha=alpha, linewidth=2, edgecolor='black')
        ax.add_patch(circle)
        
        ax.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=8, fontweight='bold')
    
    for collision in collisions:
        i, j = collision['i'], collision['j']
        pos_i, pos_j = positions[i], positions[j]
        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Particle System: {n_particles} particles, radius={radius}\n'
                f'Red=Colliding ({len(collision_pairs)} particles), Blue=Free\n'
                f'{len(collisions)} collision pairs detected')
    
    legend_elements = [
        plt.Circle((0, 0), 0.1, color='red', alpha=0.7, label='Colliding particles'),
        plt.Circle((0, 0), 0.1, color='blue', alpha=0.5, label='Free particles'),
        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5, label='Collision pairs')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig('particle_collision_system.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, ax

print(f"\n" + "="*60)
print("VISUALIZATION")
fig, ax = visualize_particles(positions, radius, collision_pairs, collisions)

if len(collisions) > 0:
    print(f"\n" + "="*60) 
    print("ADMM SOLVER TEST")
    v0 = v0_particles.flatten()
    print(f"Testing ADMM with {len(collisions)} collision constraints")
    print(f"Initial velocities shape: {v0.shape}")
    
    v_sep, z_sep, u_sep = admm_solver_separation(v0, J)
    
    print(f"\nADMM Results:")
    print(f"Constraint satisfaction: Jv = {J @ v_sep}")
    print(f"All constraints >= 0: {np.all(J @ v_sep >= -1e-10)}")
    print(f"Objective value: ||v - v0||^2 = {np.linalg.norm(v_sep - v0)**2}")
else:
    print(f"\nNo collisions to resolve with ADMM")