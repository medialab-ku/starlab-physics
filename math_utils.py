import taichi as ti

@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> float:

    ret = 0.0
    for i in a:
        ret += ti.math.dot(a[i], b[i])

    return ret

@ti.kernel
def dot2(a: ti.template(), b: ti.template()) -> float:

    ret = 0.0
    for i in a:
        ret += (a[i] * b[i])

    return ret

@ti.kernel
def max(x: ti.template()):
     for i in x:
        x[i] = ti.max(x[i], 0.0)

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

@ti.kernel
def coef_wise_op(ret: ti.template(), x: ti.template(), y: ti.template(), op: int):

    for i in x:
        if op == 0:  # mul
            ret[i] = x[i] + y[i]
        
        if op == 1:  # div
            ret[i] = x[i] / y[i]

@ti.kernel
def mean_ti(x: ti.template()) -> float:
    """Taichi kernel version of mean calculation"""
    ret = 0.0
    count = 0
    for i in x:
        ret += x[i]
        count += 1
    
    return ret / ti.max(count, 1)

def mean(x):
    """Python host function to calculate mean of a Taichi field"""
    return mean_ti(x)

def clamp(x, lo, hi):
    if x < lo: return lo
    if x > hi: return hi
    return x

def sqrt(x):
    # 음수/아주 작은 음수(라운딩) 방지
    v = float(x)
    return (0.0 if v <= 0.0 else v ** 0.5)