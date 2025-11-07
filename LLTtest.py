import taichi as ti
from taichi.linalg import SparseSolver, SparseMatrixBuilder
import numpy as np


ti.init(arch=ti.gpu)

N = 4
dim = 3
ND = N * dim
num_constraints = N - 1

# ================== Taichi version ==================

P = SparseMatrixBuilder(ND, ND, max_num_triplets=ND*ND)
A_builder = SparseMatrixBuilder(ND, ND, max_num_triplets=ND + num_constraints * dim * 4)
x = ti.Vector.field(n=dim, dtype=ti.f32, shape=N)
b = ti.Vector.field(n=dim, dtype=ti.f32, shape=N)

x_nd = ti.ndarray(dtype=ti.f32, shape=N*dim)
b_nd = ti.ndarray(dtype=ti.f32, shape=N*dim)


@ti.kernel
def build_b(b: ti.template()):
    for i in range(N):
        b[i] = ti.Vector([i, i * 2.0, (i+1)], dt=ti.f32)

@ti.kernel
def build_A(K: ti.types.sparse_matrix_builder()):
    for k in range(num_constraints):
        i = k
        j = k + 1
        for s in range(dim):
            ii = i * dim + s
            jj = j * dim + s

            K[ii, ii] += 1.0
            K[jj, jj] += 1.0
            K[ii, jj] += -1.0
            K[jj, ii] += -1.0


@ti.kernel
def build_K(A: ti.types.sparse_matrix_builder()):
    for i in range(N):
        for j in range(dim):
            row = i * dim + j
            A[row, row] += 4.0

        if i + 1 < N:
            for j in range(dim):
                row = i * dim + j
                col = (i + 1) * dim + j
                A[row, col] += 1.0
                A[col, row] += 1.0


@ti.kernel
def flatten(src: ti.template(), dst: ti.types.ndarray()):
    for i in range(N):
        for j in range(dim):
            idx = i * dim + j
            dst[idx] = src[i][j]


@ti.kernel
def unflatten(src: ti.types.ndarray(), dst: ti.template()):
    for i in range(N):
        for j in range(dim):
            idx = i * dim + j
            dst[i][j] = src[idx]


@ti.kernel
def add_mass(A: ti.types.sparse_matrix_builder(), m: ti.f32):
    for d in range(ND):
        A[d, d] += m 

# build_A(P)
add_mass(A_builder, 10.0)
build_A(A_builder)
A = A_builder.build()

build_b(b)
# A = P.build()
flatten(b, b_nd)

solver = SparseSolver(solver_type="LLT")
solver.analyze_pattern(A)
solver.factorize(A)
x_nd = solver.solve(b_nd)

unflatten(x_nd, x)

for i in range(N):
    print(f"x[{i}]: {x[i]}")
print("----------------------------------------")


# ================== Numpy version ==================

Q = np.zeros((ND, ND), dtype=np.float32)
c = np.zeros(ND, dtype=np.float32)
y = np.zeros(ND, dtype=np.float32)

for i in range(N):
    v0 = float(i)
    v1 = float(i * 2.0)
    v2 = float(i + 1)

    c[i * dim] = v0
    c[i * dim + 1] = v1
    c[i * dim + 2] = v2

for k in range(num_constraints):
    i = k
    j = k + 1
    for s in range(dim):
        ii = i * dim + s
        jj = j * dim + s

        Q[ii, ii] += 1.0
        Q[jj, jj] += 1.0
        Q[ii, jj] += -1.0
        Q[jj, ii] += -1.0


Q += 10.0 * np.eye(ND, dtype=np.float32)


# Q = L L^T
L = np.linalg.cholesky(Q)

# L @ t = c, t = L^T @ y
t = np.linalg.solve(L, c)
# L^T @ y = c 
y = np.linalg.solve(L.T, t)
y = y.reshape(N, dim)

for i in range(N):
    print(f"y[{i}]: {y[i]}")