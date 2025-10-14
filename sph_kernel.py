import taichi as ti
import numpy as np

@ti.func
def cubic_kernel(r_norm, h):
    res = ti.cast(0.0, ti.f32)

    k = 8 / np.pi
    k /= h ** 3
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1)
        else:
            res = k * 2 * ti.pow(1 - q, 3.0)
    return res


@ti.func
def cubic_kernel_derivative(r, h):

    k = 8 / np.pi
    k = 6. * k / h ** 3
    r_norm = r.norm()
    q = r_norm / h
    res = ti.Vector([0.0 for _ in range(3)])
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res


@ti.func
def spiky_kernel_derivative(r, h):

    k = 45 / (np.pi * h ** 6)
    r_norm = r.norm()

    if r_norm < 1e-6:
        r_norm = 1e-6

    grad_q = r / r_norm
    res = -k * ((h - r_norm) ** 2) * grad_q
    return res