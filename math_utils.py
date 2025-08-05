import taichi as ti

@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> float:

    ret = 0.0
    for i in a:
        ret += ti.math.dot(a[i], b[i])

    return ret

@ti.kernel
def add(ret: ti.template(), v0: ti.template(), scale: float, v1: ti.template()):
    for i in ret:
        ret[i] = v0[i] + scale * v1[i]

@ti.kernel
def scale(ret: ti.template(), scale: float, v0: ti.template()):
    for i in ret:
        ret[i] = scale * v0[i]


@ti.kernel
def inf_norm(x: ti.template()) -> float:

    ret = 0.0
    for i in x:

        tmp = x[i].norm()

        ti.atomic_max(ret, tmp)

    return ret