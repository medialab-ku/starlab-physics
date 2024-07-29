import taichi as ti
import framework.collision.distance as di
@ti.func
def __ee_st(compliance, ei_d, ei_s, edge_indices_dy, edge_indices_st, y_dy, x_st, dx_dy, nc_dy, dHat):

    v0 = edge_indices_dy[ei_d][0]
    v1 = edge_indices_dy[ei_d][1]

    v2 = edge_indices_st[ei_s][0]
    v3 = edge_indices_st[ei_s][1]

    x0, x1 = y_dy[v0], y_dy[v1]
    x2, x3 = x_st[v2], x_st[v3]

    dtype = di.d_type_EE(x0, x1, x2, x3)
    # print("ee type: ", dtype)
    g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
    d = dHat
    schur = 0.0
    if dtype == 0:
        d = di.d_PP(x0, x2)
        if d < dHat:
            g0, g2 = di.g_PP(x0, x2)
            schur = g0.dot(g0)
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)
            dx_dy[v0] += ld * g0
            nc_dy[v0] += 1

    elif dtype == 1:
        d = di.d_PP(x0, x3)
        if d < dHat:
            g0, g3 = di.g_PP(x0, x3)
            schur =  g0.dot(g0)
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)
            dx_dy[v0] +=  ld * g0
            nc_dy[v0] += 1

    elif dtype == 2:
        d = di.d_PE(x0, x2, x3)
        if d < dHat:
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            schur =  g0.dot(g0)
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)
            dx_dy[v0] +=  ld * g0
            nc_dy[v0] += 1

    elif dtype == 3:
        d = di.d_PP(x1, x2)
        if d < dHat:
            g1, g2 = di.g_PP(x1, x2)
            schur = g1.dot(g1)
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)

            dx_dy[v1] += ld * g1
            nc_dy[v1] += 1

    elif dtype == 4:
        d = di.d_PP(x1, x3)
        if d < dHat:
            g1, g3 = di.g_PP(x1, x3)
            schur = g1.dot(g1)
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)
            dx_dy[v1] += ld * g1
            nc_dy[v1] += 1

    elif dtype == 5:
        d = di.d_PE(x1, x2, x3)
        if d < dHat:
            g1, g2, g3 = di.g_PE(x1, x2, x3)
            schur = g1.dot(g1)
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)
            dx_dy[v1] += ld * g1
            nc_dy[v1] += 1

    elif dtype == 6:
        d = di.d_PE(x2, x0, x1)
        if d < dHat:
            g2, g0, g1 = di.g_PE(x2, x0, x1)
            schur = g0.dot(g0) + g1.dot(g1)
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)

            dx_dy[v0] += ld * g0
            dx_dy[v1] += ld * g1
            nc_dy[v0] += 1
            nc_dy[v1] += 1

    elif dtype == 7:
        d = di.d_PE(x3, x0, x1)
        if d < dHat:
            g3, g0, g1 = di.g_PE(x3, x0, x1)
            schur = g0.dot(g0) + g1.dot(g1)
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)

            dx_dy[v0] += ld * g0
            dx_dy[v1] += ld * g1
            nc_dy[v0] += 1
            nc_dy[v1] += 1

    elif dtype == 8:
        x01 = x0 - x1
        x23 = x2 - x3
        metric_para_EE = x01.cross(x23).norm()
        if metric_para_EE < 1e-3: print("parallel!")
        d = di.d_EE(x0, x1, x2, x3)
        if d < dHat:
            g0, g1, _, _ = di.g_EE(x0, x1, x2, x3)
            schur = ( g0.dot(g0) +
                     g1.dot(g1))
            ld = compliance * (dHat - d) / (compliance * schur + 1.0)
            dx_dy[v0] +=  ld * g0
            dx_dy[v1] += ld * g1
            nc_dy[v0] += 1
            nc_dy[v1] += 1