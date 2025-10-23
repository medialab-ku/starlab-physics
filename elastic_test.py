import numpy as np

# ---------------- helpers (names aligned to your Taichi code) ----------------
def double_dot_product(A, B):
    return float(np.tensordot(A, B))

def ssvd(F):
    # same sign handling as your Taichi code
    U, s, Vt = np.linalg.svd(F)
    V = Vt.T
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1.0
        s[2] *= -1.0
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0
        s[2] *= -1.0
    return U, np.diag(s), V

def compute_dJdF_3x3(F):
    J = np.linalg.det(F)
    return J * np.linalg.inv(F).T  # ∂J/∂F = J F^{-T}

def hessian_det_action(F, H):
    # (∂²J/∂F²)[H] = J*( tr(F^{-1}H) F^{-T} - F^{-T} H^T F^{-T} )
    J = np.linalg.det(F)
    Finv = np.linalg.inv(F)
    FinvT = Finv.T
    tr_term = np.trace(Finv @ H)
    return J * (tr_term * FinvT - FinvT @ H.T @ FinvT)

def hessian_det_matrix(F):
    H9 = np.zeros((9, 9))
    basis = []
    for a in range(3):
        for b in range(3):
            E = np.zeros((3, 3)); E[a, b] = 1.0
            basis.append(E)
    for j, Ej in enumerate(basis):
        col = hessian_det_action(F, Ej).reshape(-1)
        H9[:, j] = col
    return H9

def initialize_D():
    # exactly your D[0..5]
    D = []
    D.append(np.array([[0,0,0],[0,0,1],[0,-1,0]], float))
    D.append(np.array([[0,0,1],[0,0,0],[-1,0,0]], float))
    D.append(np.array([[0,1,0],[-1,0,0],[0,0,0]], float))
    D.append(np.array([[0,0,0],[0,0,1],[0,1,0]], float))
    D.append(np.array([[0,0,1],[0,0,0],[1,0,0]], float))
    D.append(np.array([[0,1,0],[1,0,0],[0,0,0]], float))
    return D

def eq33_diagonal_eigs(I_C, J):
    eps = 1e-12
    ICc = max(I_C, eps)
    t = 2.0 * np.sqrt(ICc / 3.0)
    u = (3.0 * J / ICc) * np.sqrt(3.0 / ICc)
    arg = np.clip(u, -1.0, 1.0)
    the0 = np.arccos(arg)
    return np.array([t * np.cos((the0 + 2.0 * np.pi * k) / 3.0) for k in range(3)])

def build_Q_lamb_like_taichi(F):
    """
    Reproduce your compute_P() logic for Q[0..8], lamb[0..8]
      - ssvd sign handling (may yield negative s2)
      - lamb[3:6] = s0,s1,s2  ; lamb[6:9] = -s0,-s1,-s2
      - Q[3..8]   = 1/sqrt(2) * U D[k] V^T    (k=0..5)
      - lamb[0..2] from Eq.(33)
      - Q[0..2]   via D_k = diag(a*c + b*λ,  b*c + a*λ,  λ^2 - c^2), cyclic (a,b,c)=(s_k,s_{k+1},s_{k+2})
        and Frobenius normalization
    """
    U, sigM, V = ssvd(F)
    s0, s1, s2 = sigM[0,0], sigM[1,1], sigM[2,2]
    sigv = np.array([s0, s1, s2], float)
    I_C  = float(np.sum(sigv * sigv))
    J    = float(np.linalg.det(F))

    Q = np.zeros((9, 3, 3), dtype=float)
    lamb = np.zeros(9, dtype=float)

    # off-diagonal 6 modes
    D = initialize_D()
    for k in range(6):
        Q[3 + k] = (1.0 / np.sqrt(2.0)) * (U @ D[k] @ V.T)
    lamb[3:6] = sigv
    lamb[6:9] = -sigv

    # diagonal 3 modes (Eq.33 + your D_k construction)
    lam_diag = eq33_diagonal_eigs(I_C, J)
    for k in range(3):
        a = sigv[k]
        b = sigv[(k + 1) % 3]
        c = sigv[(k + 2) % 3]
        lamk = lam_diag[k]
        lamb[k] = lamk
        Dk = np.diag([a * c + b * lamk, b * c + a * lamk, lamk * lamk - c * c])
        qn = np.linalg.norm(Dk)
        if qn < 1e-12:
            Q[k] = np.eye(3)
        else:
            Q[k] = (U @ Dk @ V.T) / qn

    return Q, lamb

def spectral_sum_matrix(Q, lamb):
    R = np.zeros((9, 9), dtype=float)
    for k in range(9):
        v = Q[k].reshape(-1)
        R += lamb[k] * np.outer(v, v)
    return R

# ---------------- tetra example: build F ----------------
# Rest tetra:  (0,0,0), (1,0,0), (0,1,0), (0,0,1)
# Deformed:    (0,0,0), (2,0,0), (0,0.5,0), (0,0,2)
# For this canonical tetra, Dm = I ⇒ F = Ds
F = np.diag([2.0, -0.5, 2.0])

# ---------------- compare LEFT vs RIGHT(paper) ----------------
H_left = hessian_det_matrix(F)             # analytic ∂²J/∂F²
Q, lamb = build_Q_lamb_like_taichi(F)      # your Taichi-style eigensystem
H_right_paper = spectral_sum_matrix(Q, lamb)

# sanity: numeric spectral of H_left
w_num, V_num = np.linalg.eigh(H_left)
H_right_numeric = (V_num * w_num) @ V_num.T

np.set_printoptions(precision=6, suppress=True)
print("F =\n", F)
print("J =", np.linalg.det(F))

print("\nLEFT  (analytic Hessian 9x9) =\n", H_left)
print("\nRIGHT (paper-coded spectral) = Σ λ_k vec(Q_k) vec(Q_k)^T\n", H_right_paper)
print("\nRIGHT (numeric spectral)     = V diag(w) V^T\n", H_right_numeric)

print("\n‖LEFT - RIGHT(paper )‖_F   =", np.linalg.norm(H_left - H_right_paper))
print("‖LEFT - RIGHT(numeric)‖_F =", np.linalg.norm(H_left - H_right_numeric))

# operator action test on random F_tmp (like your F_tmp direction)
rng = np.random.default_rng(0)
F_tmp = rng.standard_normal((3,3))
left_act  = hessian_det_action(F, F_tmp)
right_act = np.zeros((3,3))
for k in range(9):
    qk = Q[k]
    right_act += lamb[k] * double_dot_product(qk, F_tmp) * qk

right_num_act = (V_num @ (w_num * (V_num.T @ F_tmp.reshape(-1)))).reshape(3,3)

print("\nTest on random F_tmp:")
print("LEFT  action:\n", left_act)
print("RIGHT paper  action:\n", right_act)
print("RIGHT numeric action:\n", right_num_act)
print("‖LEFT - RIGHT(paper )‖_F =", np.linalg.norm(left_act - right_act))
print("‖LEFT - RIGHT(numeric)‖_F =", np.linalg.norm(left_act - right_num_act))
