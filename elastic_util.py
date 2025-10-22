# The functions for elastic deformation
# dPsidF : first derivative of Psi w.r.t F
# d2PsidF2 : second derivative of Psi w.r.t F

import taichi as ti

@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V

@ti.func
def compute_dFdx_T_N(B, N):

    # vec(dFdx)^T vec(N) ， N:3x3(before vec), B 3x3 -> 12x1
    B00, B01, B02,  B10, B11, B12, B20, B21, B22 = B[0,0], B[0,1],  B[0,2],  B[1,0],  B[1,1],  B[1,2],  B[2,0],  B[2,1],  B[2,2]
    N00, N01,  N02,  N10,  N11,  N12,  N20,  N21,  N22 = N[0,0], N[0,1], N[0,2], N[1,0], N[1,1], N[1,2], N[2,0], N[2,1], N[2,2]
    r3 = B00 * N00 + B01 * N01 + B02 * N02
    r4 = B00 * N10 + B01 * N11 + B02 * N12
    r5 = B00 * N20 + B01 * N21 + B02 * N22
    r6 = B10 * N00 + B11 * N01 + B12 * N02
    r7 = B10 * N10 + B11 * N11 + B12 * N12
    r8 = B10 * N20 + B11 * N21 + B12 * N22
    r9 = B20 * N00 + B21 * N01 + B22 * N02
    r10 = B20 * N10 + B21 * N11 + B22 * N12
    r11 = B20 * N20 + B21 * N21 + B22 * N22
    r0 = -r3 - r6 - r9
    r1 = -r4 - r7 - r10
    r2 = -r5 - r8 - r11
    return ti.Matrix([r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11],float)

@ti.func
def compute_dFdxT_p(B, p):
    # dFdx transpose 12x9, p 9x1 -> 12x1
    B00, B01, B02, B10, B11, B12, B20, B21, B22 = B[0, 0], B[0, 1], B[0, 2], B[1, 0], B[1, 1], B[1, 2], B[2, 0], B[2, 1], B[2, 2]
    p0, p1, p2, p3, p4, p5, p6, p7, p8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
    r3 = B00 * p0 + B01 * p3 + B02 * p6
    r4 = B00 * p1 + B01 * p4 + B02 * p7
    r5 = B00 * p2 + B01 * p5 + B02 * p8
    r6 = B10 * p0 + B11 * p3 + B12 * p6
    r7 = B10 * p1 + B11 * p4 + B12 * p7
    r8 = B10 * p2 + B11 * p5 + B12 * p8
    r9 = B20 * p0 + B21 * p3 + B22 * p6
    r10 = B20 * p1 + B21 * p4 + B22 * p7
    r11 = B20 * p2 + B21 * p5 + B22 * p8
    r0 = -(r3 + r6 + r9)
    r1 = -(r4 + r7 + r10)
    r2 = -(r5 + r8 + r11)
    return ti.Matrix([r0, r1, r2,
                      r3, r4, r5,
                      r6, r7, r8,
                      r9, r10, r11], float)

@ti.func
def compute_dFdx_p(B, p):
    # dFdx 9x12, p 12x1
    B00, B01, B02, B10, B11, B12, B20, B21, B22 = B[0, 0], B[0, 1], B[0, 2], B[1, 0], B[1, 1], B[1, 2], B[2, 0], B[2, 1], B[2, 2]
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]
    t1 = - B00 - B10 - B20
    t2 = - B01 - B11 - B21
    t3 = - B02 - B12 - B22
    return ti.Matrix([B00*p3 + B10*p6 + B20*p9 + p0*t1,
                        B00*p4 + B10*p7 + B20*p10 + p1*t1,
                        B00*p5 + B10*p8 + B20*p11 + p2*t1,
                        B01*p3 + B11*p6 + B21*p9 + p0*t2,
                        B01*p4 + B11*p7 + B21*p10 + p1*t2,
                        B01*p5 + B11*p8 + B21*p11 + p2*t2,
                        B02*p3 + B12*p6 + B22*p9 + p0*t3,
                        B02*p4 + B12*p7 + B22*p10 + p1*t3,
                        B02*p5 + B12*p8 + B22*p11 + p2*t3],float)

@ti.func
def compute_diag_h4(la0,la1,la2, B, U, V):
    B00, B01, B02,  B10, B11, B12, B20, B21, B22 = B[0,0], B[0,1],  B[0,2],  B[1,0],  B[1,1],  B[1,2],  B[2,0],  B[2,1],  B[2,2]
    U00, U01, U02,  U10, U11, U12, U20, U21, U22 = U[0,0], U[0,1],  U[0,2],  U[1,0],  U[1,1],  U[1,2],  U[2,0],  U[2,1],  U[2,2]
    V00, V01, V02,  V10, V11, V12, V20, V21, V22 = V[0,0], V[0,1],  V[0,2],  V[1,0],  V[1,1],  V[1,2],  V[2,0],  V[2,1],  V[2,2]
    r0 = la0*((U00*V01 - U01*V00)*(B00 + B10 + B20) + (U00*V11 - U01*V10)*(B01 + B11 + B21) + (U00*V21 - U01*V20)*(B02 + B12 + B22))**2 + la1*((U01*V02 - U02*V01)*(B00 + B10 + B20) + (U01*V12 - U02*V11)*(B01 + B11 + B21) + (U01*V22 - U02*V21)*(B02 + B12 + B22))**2 + la2*((U00*V02 - U02*V00)*(B00 + B10 + B20) + (U00*V12 - U02*V10)*(B01 + B11 + B21) + (U00*V22 - U02*V20)*(B02 + B12 + B22))**2
    r1 = la0*((U10*V01 - U11*V00)*(B00 + B10 + B20) + (U10*V11 - U11*V10)*(B01 + B11 + B21) + (U10*V21 - U11*V20)*(B02 + B12 + B22))**2 + la1*((U11*V02 - U12*V01)*(B00 + B10 + B20) + (U11*V12 - U12*V11)*(B01 + B11 + B21) + (U11*V22 - U12*V21)*(B02 + B12 + B22))**2 + la2*((U10*V02 - U12*V00)*(B00 + B10 + B20) + (U10*V12 - U12*V10)*(B01 + B11 + B21) + (U10*V22 - U12*V20)*(B02 + B12 + B22))**2
    r2 = la0*((U20*V01 - U21*V00)*(B00 + B10 + B20) + (U20*V11 - U21*V10)*(B01 + B11 + B21) + (U20*V21 - U21*V20)*(B02 + B12 + B22))**2 + la1*((U21*V02 - U22*V01)*(B00 + B10 + B20) + (U21*V12 - U22*V11)*(B01 + B11 + B21) + (U21*V22 - U22*V21)*(B02 + B12 + B22))**2 + la2*((U20*V02 - U22*V00)*(B00 + B10 + B20) + (U20*V12 - U22*V10)*(B01 + B11 + B21) + (U20*V22 - U22*V20)*(B02 + B12 + B22))**2
    r3 = la0*(B00*(U00*V01 - U01*V00) + B01*(U00*V11 - U01*V10) + B02*(U00*V21 - U01*V20))**2 + la1*(B00*(U01*V02 - U02*V01) + B01*(U01*V12 - U02*V11) + B02*(U01*V22 - U02*V21))**2 + la2*(B00*(U00*V02 - U02*V00) + B01*(U00*V12 - U02*V10) + B02*(U00*V22 - U02*V20))**2
    r4 = la0*(B00*(U10*V01 - U11*V00) + B01*(U10*V11 - U11*V10) + B02*(U10*V21 - U11*V20))**2 + la1*(B00*(U11*V02 - U12*V01) + B01*(U11*V12 - U12*V11) + B02*(U11*V22 - U12*V21))**2 + la2*(B00*(U10*V02 - U12*V00) + B01*(U10*V12 - U12*V10) + B02*(U10*V22 - U12*V20))**2
    r5 = la0*(B00*(U20*V01 - U21*V00) + B01*(U20*V11 - U21*V10) + B02*(U20*V21 - U21*V20))**2 + la1*(B00*(U21*V02 - U22*V01) + B01*(U21*V12 - U22*V11) + B02*(U21*V22 - U22*V21))**2 + la2*(B00*(U20*V02 - U22*V00) + B01*(U20*V12 - U22*V10) + B02*(U20*V22 - U22*V20))**2
    r6 = la0*(B10*(U00*V01 - U01*V00) + B11*(U00*V11 - U01*V10) + B12*(U00*V21 - U01*V20))**2 + la1*(B10*(U01*V02 - U02*V01) + B11*(U01*V12 - U02*V11) + B12*(U01*V22 - U02*V21))**2 + la2*(B10*(U00*V02 - U02*V00) + B11*(U00*V12 - U02*V10) + B12*(U00*V22 - U02*V20))**2
    r7 = la0*(B10*(U10*V01 - U11*V00) + B11*(U10*V11 - U11*V10) + B12*(U10*V21 - U11*V20))**2 + la1*(B10*(U11*V02 - U12*V01) + B11*(U11*V12 - U12*V11) + B12*(U11*V22 - U12*V21))**2 + la2*(B10*(U10*V02 - U12*V00) + B11*(U10*V12 - U12*V10) + B12*(U10*V22 - U12*V20))**2
    r8 = la0*(B10*(U20*V01 - U21*V00) + B11*(U20*V11 - U21*V10) + B12*(U20*V21 - U21*V20))**2 + la1*(B10*(U21*V02 - U22*V01) + B11*(U21*V12 - U22*V11) + B12*(U21*V22 - U22*V21))**2 + la2*(B10*(U20*V02 - U22*V00) + B11*(U20*V12 - U22*V10) + B12*(U20*V22 - U22*V20))**2
    r9 = la0*(B20*(U00*V01 - U01*V00) + B21*(U00*V11 - U01*V10) + B22*(U00*V21 - U01*V20))**2 + la1*(B20*(U01*V02 - U02*V01) + B21*(U01*V12 - U02*V11) + B22*(U01*V22 - U02*V21))**2 + la2*(B20*(U00*V02 - U02*V00) + B21*(U00*V12 - U02*V10) + B22*(U00*V22 - U02*V20))**2
    r10 = la0*(B20*(U10*V01 - U11*V00) + B21*(U10*V11 - U11*V10) + B22*(U10*V21 - U11*V20))**2 + la1*(B20*(U11*V02 - U12*V01) + B21*(U11*V12 - U12*V11) + B22*(U11*V22 - U12*V21))**2 + la2*(B20*(U10*V02 - U12*V00) + B21*(U10*V12 - U12*V10) + B22*(U10*V22 - U12*V20))**2
    r11 = la0*(B20*(U20*V01 - U21*V00) + B21*(U20*V11 - U21*V10) + B22*(U20*V21 - U21*V20))**2 + la1*(B20*(U21*V02 - U22*V01) + B21*(U21*V12 - U22*V11) + B22*(U21*V22 - U22*V21))**2 + la2*(B20*(U20*V02 - U22*V00) + B21*(U20*V12 - U22*V10) + B22*(U20*V22 - U22*V20))**2
    return 0.5 * ti.Vector([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11],float)

@ti.func
def compute_pT_h4_p(la0,la1,la2,U,V,d):
    #d = dFdx @ p 9x1
    U00, U01, U02,  U10, U11, U12, U20, U21, U22 = U[0,0], U[0,1],  U[0,2],  U[1,0],  U[1,1],  U[1,2],  U[2,0],  U[2,1],  U[2,2]
    V00, V01, V02,  V10, V11, V12, V20, V21, V22 = V[0,0], V[0,1],  V[0,2],  V[1,0],  V[1,1],  V[1,2],  V[2,0],  V[2,1],  V[2,2]
    d0, d1, d2, d3, d4, d5, d6, d7, d8 = d[0], d[1],  d[2],  d[3],  d[4],  d[5],  d[6],  d[7],  d[8]
    return 0.5 * (
            la0*( d0*(U00*V01 - U01*V00) + d1*(U10*V01 - U11*V00) + d2*(U20*V01 - U21*V00) + d3*(U00*V11 - U01*V10) + d4*(U10*V11 - U11*V10) + d5*(U20*V11 - U21*V10) + d6*(U00*V21 - U01*V20) + d7*(U10*V21 - U11*V20) + d8*(U20*V21 - U21*V20) )**2 +
            la1*(d0*(U01*V02 - U02*V01) + d1*(U11*V02 - U12*V01) + d2*(U21*V02 - U22*V01) + d3*(U01*V12 - U02*V11) + d4*(U11*V12 - U12*V11) + d5*(U21*V12 - U22*V11) + d6*(U01*V22 - U02*V21) + d7*(U11*V22 - U12*V21) + d8*(U21*V22 - U22*V21))**2 +
            la2*(d0*(U00*V02 - U02*V00) + d1*(U10*V02 - U12*V00) + d2*(U20*V02 - U22*V00) + d3*(U00*V12 - U02*V10) + d4*(U10*V12 - U12*V10) + d5*(U20*V12 - U22*V10) + d6*(U00*V22 - U02*V20) + d7*(U10*V22 - U12*V20) + d8*(U20*V22 - U22*V20))**2
    )

@ti.func
def compute_diag_dFdx_T_dFdx(B):
    B00, B01, B02,  B10, B11, B12, B20, B21, B22 = B[0,0], B[0,1],  B[0,2],  B[1,0],  B[1,1],  B[1,2],  B[2,0],  B[2,1],  B[2,2]
    tmp0 = (B00 + B10 + B20)**2 + (B01 + B11 + B21)**2 + (B02 + B12 + B22)**2
    tmp1 = B00**2 + B01**2 + B02**2
    tmp2 = B10**2 + B11**2 + B12**2
    tmp3 = B20**2 + B21**2 + B22**2
    return ti.Vector([tmp0, tmp0, tmp0, tmp1, tmp1, tmp1, tmp2, tmp2, tmp2, tmp3, tmp3, tmp3], float)

@ti.func
def flatten_matrix(A):
    # flattern 3x3 A to 9x1 vector (column first)
    return ti.Vector([A[0,0], A[1,0], A[2,0], A[0,1], A[1,1],A[2,1],A[0,2],A[1,2],A[2,2]],float)

@ti.func
def compute_d_H3_d(F, d):
    # d: 9x1, H3 9x9
    d0, d1, d2, d3, d4, d5, d6, d7, d8= d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]
    F00, F01, F02, F10, F11, F12, F20, F21, F22 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
    return 2 * (F00 * d4 * d8 - F00 * d5 * d7 - F01 * d1 * d8 + F01 * d2 * d7
                + F02 * d1 * d5 - F02 * d2 * d4 - F10 * d3 * d8 + F10 * d5 * d6
                + F11 * d0 * d8 - F11 * d2 * d6 - F12 * d0 * d5 + F12 * d2 * d3
                + F20 * d3 * d7 - F20 * d4 * d6 - F21 * d0 * d7 + F21 * d1 * d6
                + F22 * d0 * d4 - F22 * d1 * d3)

@ti.func
def compute_dJdF_3x3(F):
    """F:3x3->dJdF:3x3(before vec)"""
    F00, F01, F02, F10, F11, F12, F20, F21, F22 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
    return ti.Matrix([[F11 * F22 - F12 * F21, -F10 * F22 + F12 * F20, F10 * F21 - F11 * F20],
            [-F01 * F22 + F02 * F21, F00 * F22 - F02 * F20, -F00 * F21 + F01 * F20],
            [F01 * F12 - F02 * F11, -F00 * F12 + F02 * F10, F00 * F11 - F01 * F10]], float)

@ti.func
def compute_vec_dJdF(F):
    """F:3x3->dJdF:9x1(after vec)"""
    F00, F01, F02, F10, F11, F12, F20, F21, F22 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
    return ti.Matrix([F11 * F22 - F12 * F21, -F01 * F22 + F02 * F21, F01 * F12 - F02 * F11,
                      -F10 * F22 + F12 * F20, F00 * F22 - F02 * F20, -F00 * F12 + F02 * F10,
                      F10 * F21 - F11 * F20, -F00 * F21 + F01 * F20, F00 * F11 - F01 * F10],float)

@ti.func
def compute_H3(F):
    """F:3x3->H3:9x9"""
    F00, F01, F02, F10, F11, F12, F20, F21, F22 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
    return ti.Matrix([[0, 0, 0, 0, F22, -F12, 0, -F21, F11],
                      [0, 0, 0, -F22, 0, F02, F21, 0, -F01],
                      [0, 0, 0, F12, -F02, 0, -F11, F01, 0],
                      [0, -F22, F12, 0, 0, 0, 0, F20, -F10],
                      [F22, 0, -F02, 0, 0, 0, -F20, 0, F00],
                      [-F12, F02, 0, 0, 0, 0, F10, -F00, 0],
                      [0, F21, -F11, 0, -F20, F10, 0, 0, 0],
                      [-F21, 0, F01, F20, 0, -F00, 0, 0, 0],
                      [F11, -F01, 0, -F10, F00, 0, 0, 0, 0]],float)


@ti.func
def compute_dtdx_t(t, cord):
    t0 = t[0]
    t1 = t[1]
    t2 = t[2]
    c0 = cord[0]
    c1 = cord[1]
    c2 = cord[2]
    c3 = cord[3]
    return ti.Vector([c0 * t0, c0 * t1, c0 * t2, c1 * t0, c1 * t1, c1 * t2, c2 * t0, c2 * t1, c2 * t2, c3 * t0, c3 * t1, c3 * t2],float)

@ti.func
def compute_d_dtdx(d, cord):
    return cord[0] * d[0:3] + cord[1] * d[3:6] + cord[2] * d[6:9] + cord[3] * d[9:12]

@ti.func
def compute_dFdx(DmInv):
    dFdx = ti.Matrix.zero(float,9,12)
    m = DmInv[0,0]
    n = DmInv[0,1]
    o = DmInv[0,2]
    p = DmInv[1,0]
    q = DmInv[1,1]
    r = DmInv[1,2]
    s = DmInv[2,0]
    t = DmInv[2,1]
    u = DmInv[2,2]
    t1 = -m-p-s
    t2 = -n-q-t
    t3 = -o-r-u
    dFdx[0,0] = t1
    dFdx[0,3] = m
    dFdx[0,6] = p
    dFdx[0,9] = s

    dFdx[1,1] = t1
    dFdx[1,4] = m
    dFdx[1,7] = p
    dFdx[1,10] = s

    dFdx[2,2] = t1
    dFdx[2,5] = m
    dFdx[2,8] = p
    dFdx[2,11] = s

    dFdx[3,0] = t2
    dFdx[3,3] = n
    dFdx[3,6] = q
    dFdx[3,9] = t

    dFdx[4,1] = t2
    dFdx[4,4] = n
    dFdx[4,7] = q
    dFdx[4,10] = t

    dFdx[5,2] = t2
    dFdx[5,5] = n
    dFdx[5,8] = q
    dFdx[5,11] = t

    dFdx[6,0] = t3
    dFdx[6,3] = o
    dFdx[6,6] = r
    dFdx[6,9] = u

    dFdx[7,1] = t3
    dFdx[7,4] = o
    dFdx[7,7] = r
    dFdx[7,10] = u

    dFdx[8,2] = t3
    dFdx[8,5] = o
    dFdx[8,8] = r
    dFdx[8,11] = u
    return dFdx

# ARAP
@ti.func
def compute_Psi_ARAP(F, mu, la):
    U, sig, V = ssvd(F)
    R = U @ (V.transpose())
    Psi = mu * (F - R).norm_sqr()
    return Psi

@ti.func
def compute_dPsidF_ARAP(F, mu, la):
    U, sig, V = ssvd(F)
    R = U @ (V.transpose())
    dPsidF_3x3 = 2.0 * mu * (F - R)
    return dPsidF_3x3

@ti.func
def compute_dPsidx_ARAP(F, B, mu, la):
    dPsidF = compute_dPsidF_ARAP(F, mu, la)  # 3x3
    dPsidx = compute_dFdx_T_N(B, dPsidF)  # 12x1
    return dPsidx

@ti.func
def compute_diag_d2Psidx2_ARAP(F, B, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    W0 = compute_dFdx_T_N(B, Q0) ** 2
    W1 = compute_dFdx_T_N(B, Q1) ** 2
    W2 = compute_dFdx_T_N(B, Q2) ** 2
    diag_h4 = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
    X = compute_diag_dFdx_T_dFdx(B)
    return mu * (2.0 * X - diag_h4)


@ti.func
def compute_pHp_ARAP(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = dFdx_p.norm_sqr()
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    # remove 1/sqrt(2) here
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    Q0_flatten = flatten_matrix(Q0)
    Q1_flatten = flatten_matrix(Q1)
    Q2_flatten = flatten_matrix(Q2)
    ret1 = lambda0 * (Q0_flatten.dot(dFdx_p)) ** 2 + lambda1 * (Q1_flatten.dot(dFdx_p)) ** 2 + lambda2 * (
        Q2_flatten.dot(dFdx_p)) ** 2
    ret = mu * (2.0 * ret0 - ret1)
    return ret


# ARAP filter
@ti.func
def compute_diag_d2Psidx2_ARAP_filter(F, B, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    W0 = compute_dFdx_T_N(B, Q0) ** 2
    W1 = compute_dFdx_T_N(B, Q1) ** 2
    W2 = compute_dFdx_T_N(B, Q2) ** 2
    diag_h4 = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
    X = compute_diag_dFdx_T_dFdx(B)
    return mu * (2.0 * X - diag_h4)


@ti.func
def compute_pHp_ARAP_filter(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = dFdx_p.norm_sqr()
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    # remove 1/sqrt(2) here
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    Q0_flatten = flatten_matrix(Q0)
    Q1_flatten = flatten_matrix(Q1)
    Q2_flatten = flatten_matrix(Q2)
    ret1 = lambda0 * (Q0_flatten.dot(dFdx_p)) ** 2 + lambda1 * (Q1_flatten.dot(dFdx_p)) ** 2 + lambda2 * (
        Q2_flatten.dot(dFdx_p)) ** 2
    ret = mu * (2.0 * ret0 - ret1)
    return ret


# FCR
@ti.func
def compute_Psi_FCR(F, mu, la):
    U, sig, V = ssvd(F)
    R = U @ (V.transpose())
    J = F.determinant()
    Psi = mu * (F - R).norm_sqr() + 0.5 * la * (J - 1.0) ** 2
    return Psi


@ti.func
def compute_dPsidF_FCR(F, mu, la):
    U, sig, V = ssvd(F)
    R = U @ (V.transpose())
    dPsidF_3x3_ARAP = 2.0 * mu * (F - R)
    J = F.determinant()
    dJdF = compute_dJdF_3x3(F)
    dPsidF_3x3 = dPsidF_3x3_ARAP + la * (J - 1) * dJdF
    return dPsidF_3x3


@ti.func
def compute_dPsidx_FCR(F, B, mu, la):
    dPsidF = compute_dPsidF_FCR(F, mu, la)  # 3x3
    dPsidx = compute_dFdx_T_N(B, dPsidF)
    return dPsidx


@ti.func
def compute_diag_d2Psidx2_FCR(F, B, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    W0 = compute_dFdx_T_N(B, Q0) ** 2
    W1 = compute_dFdx_T_N(B, Q1) ** 2
    W2 = compute_dFdx_T_N(B, Q2) ** 2
    diag_h4 = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
    X = compute_diag_dFdx_T_dFdx(B)
    diag_ARAP = mu * (2.0 * X - diag_h4)
    g3 = compute_dJdF_3x3(F)  # g3 = dJdF
    dFdx_T_g3 = compute_dFdx_T_N(B, g3)
    diagJ = la * dFdx_T_g3 ** 2
    return diag_ARAP + diagJ


@ti.func
def compute_diag_d2Psidx2_FCR_filter(F, B, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    W0 = compute_dFdx_T_N(B, Q0) ** 2
    W1 = compute_dFdx_T_N(B, Q1) ** 2
    W2 = compute_dFdx_T_N(B, Q2) ** 2
    diag_h4 = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
    X = compute_diag_dFdx_T_dFdx(B)
    diag_ARAP = mu * (2.0 * X - diag_h4)
    g3 = compute_dJdF_3x3(F)  # g3 = dJdF
    dFdx_T_g3 = compute_dFdx_T_N(B, g3)
    diagJ = la * dFdx_T_g3 ** 2
    return diag_ARAP + diagJ


@ti.func
def compute_pHp_FCR(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = dFdx_p.norm_sqr()
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    # remove 1/sqrt(2) here
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    Q0_flatten = flatten_matrix(Q0)
    Q1_flatten = flatten_matrix(Q1)
    Q2_flatten = flatten_matrix(Q2)
    ret1 = lambda0 * (Q0_flatten.dot(dFdx_p)) ** 2 + lambda1 * (Q1_flatten.dot(dFdx_p)) ** 2 + lambda2 * (
        Q2_flatten.dot(dFdx_p)) ** 2
    ret_ARAP = mu * (2.0 * ret0 - ret1)
    g_3 = compute_vec_dJdF(F)
    ret_2 = la * ((g_3.dot(dFdx_p)) ** 2)
    J = F.determinant()
    ret_3 = la * (J - 1.0) * compute_d_H3_d(F, dFdx_p)
    return ret_ARAP + ret_2 + ret_3


@ti.func
def compute_pHp_FCR_filter(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = dFdx_p.norm_sqr()
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0, U1, U2 = U[:, 0], U[:, 1], U[:, 2]
    V0, V1, V2 = V[:, 0], V[:, 1], V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    # remove 1/sqrt(2) here
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    Q0_flatten = flatten_matrix(Q0)
    Q1_flatten = flatten_matrix(Q1)
    Q2_flatten = flatten_matrix(Q2)
    ret1 = lambda0 * (Q0_flatten.dot(dFdx_p)) ** 2 + lambda1 * (Q1_flatten.dot(dFdx_p)) ** 2 + lambda2 * (
        Q2_flatten.dot(dFdx_p)) ** 2
    ret_ARAP = mu * (2.0 * ret0 - ret1)
    g_3 = compute_vec_dJdF(F)
    ret_2 = la * ((g_3.dot(dFdx_p)) ** 2)
    J = F.determinant()
    ret_3 = la * (J - 1.0) * compute_d_H3_d(F, dFdx_p)
    if ret_3 < 0.0:
        ret_3 = 0.0
    return ret_ARAP + ret_2 + ret_3


# SNH
@ti.func
def compute_Psi_SNH(F, mu, la):
    J = F.determinant()
    return 0.5 * mu * (F.norm_sqr() - 3) - mu * (J - 1) + 0.5 * la * (J - 1) ** 2


@ti.func
def compute_dPsidF_SNH(F, mu, la):
    # return 3x3
    J = F.determinant()
    dJdF = compute_dJdF_3x3(F)
    dPsidF = mu * F + (- mu + la * (J - 1)) * dJdF
    return dPsidF


@ti.func
def compute_dPsidx_SNH(F, B, mu, la):
    dPsidF = compute_dPsidF_SNH(F, mu, la)  # 3x3
    dPsidx = compute_dFdx_T_N(B, dPsidF)  # 12x1
    return dPsidx


@ti.func
def compute_diag_d2Psidx2_SNH(F, B, mu, la):
    g3 = compute_dJdF_3x3(F)  # g3 = dJdF
    dFdx_T_g3 = compute_dFdx_T_N(B, g3)
    diag1 = la * dFdx_T_g3 ** 2
    diag2 = mu * compute_diag_dFdx_T_dFdx(B)
    return diag1 + diag2


@ti.func
def compute_pHp_SNH(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = mu * dFdx_p.norm_sqr()
    g_3 = compute_vec_dJdF(F)
    ret1 = la * ((g_3.dot(dFdx_p)) ** 2)
    J = F.determinant()
    ret2 = (la * (J - 1.0) - mu) * compute_d_H3_d(F, dFdx_p)
    return ret0 + ret1 + ret2

@ti.func
def compute_Psi_NH(F, mu, la):
    J = F.determinant()
    return 0.5 * mu * (F.norm_sqr() - 3) - mu * ti.log(J) + 0.5 * la * ti.log(J) ** 2


@ti.func
def compute_dPsidF_NH(F, mu, la):
    # return 3x3
    J = F.determinant()
    dJdF = compute_dJdF_3x3(F)
    para = (- mu + la * ti.log(J)) / J
    dPsidF = mu * F + para * dJdF
    return dPsidF.transpose()


@ti.func
def compute_dPsidx_NH(F, B, mu, la):
    dPsidF = compute_dPsidF_SNH(F, mu, la)  # 3x3
    dPsidx = compute_dFdx_T_N(B, dPsidF)  # 12x1
    return dPsidx


@ti.func
def compute_diag_d2Psidx2_NH(F, B, mu, la):
    J = F.determinant()
    g3 = compute_dJdF_3x3(F)  # g3 = dJdF
    dFdx_T_g3 = compute_dFdx_T_N(B, g3)
    para = (la * (1.0 - ti.log(J)) + mu) / (J * J)
    diag1 = para * dFdx_T_g3 ** 2
    diag2 = mu * compute_diag_dFdx_T_dFdx(B)
    return diag1 + diag2


@ti.func
def compute_pHp_NH(F, B, p, mu, la):
    dFdx_p = compute_dFdx_p(B, p)  # 9x1
    ret0 = mu * dFdx_p.norm_sqr()
    g_3 = compute_vec_dJdF(F)
    J = F.determinant()
    para1 = (la * (1.0 - ti.log(J)) + mu) / (J * J)
    ret1 = para1 * ((g_3.dot(dFdx_p)) ** 2)
    para2 = (la * ti.log(J) - mu) / J
    ret2 = para2 * compute_d_H3_d(F, dFdx_p)
    return ret0 + ret1 + ret2


@ti.func
def compute_d2PsidF2_ARAP(F, mu, la):
    U,sig,V = ssvd(F)
    s0 = sig[0,0]
    s1 = sig[1,1]
    s2 = sig[2,2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    q0 = flatten_matrix(Q0)
    # remove 1/sqrt(2) here
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    d2PsidF2 = - mu *( lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)) )
    # remove 2.0 here
    for i in ti.static(range(9)):
        d2PsidF2[i,i] += 2 * mu
    return d2PsidF2

@ti.func
def compute_d2PsidF2_ARAP_filter(F, mu, la):
    U,sig,V = ssvd(F)
    s0 = sig[0,0]
    s1 = sig[1,1]
    s2 = sig[2,2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0

    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    q0 = flatten_matrix(Q0)
    # remove 1/sqrt(2) here
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    d2PsidF2 = - mu * ( lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)) )
    # remove 2.0 here
    for i in ti.static(range(9)):
        d2PsidF2[i,i] += 2.0 * mu
    return d2PsidF2

@ti.func
def compute_d2PsidF2_SNH(F, mu, la):
    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()
    ret = mu * g3.outer_product(g3) + (la * (J - 1.0) - mu) * H3
    for i in range(9):
        ret[i,i] += mu
    return ret

@ti.func
def compute_d2PsidF2_NH(F, mu, la):
    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()
    para1 = (la * (1.0 - ti.log(J)) + mu) / (J * J)
    para2 = (la * ti.log(J) - mu) / J
    ret = para1 * g3.outer_product(g3) + para2 * H3
    for i in range(9):
        ret += mu
    return ret

@ti.func
def compute_d2PsidF2_FCR(F, mu, la):
    U,sig,V = ssvd(F)
    s0 = sig[0,0]
    s1 = sig[1,1]
    s2 = sig[2,2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    q0 = flatten_matrix(Q0)
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    # ARAP
    d2PsidF2_ARAP = - mu *( lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)) )
    for i in ti.static(range(9)):
        d2PsidF2_ARAP[i,i] += 2 * mu

    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()
    d2PsidF2_FCR = d2PsidF2_ARAP + la * g3.outer_product(g3) + la * (J-1) * H3
    return d2PsidF2_FCR


@ti.func
def compute_d2PsidF2_FCR_filter(F, mu, la):
    U, sig, V = ssvd(F)
    s0 = sig[0, 0]
    s1 = sig[1, 1]
    s2 = sig[2, 2]
    lambda0 = 2.0 / (s1 + s2)
    lambda1 = 2.0 / (s0 + s2)
    lambda2 = 2.0 / (s0 + s1)
    if s1 + s2 < 2.0:
        lambda0 = 1.0
    if s0 + s2 < 2.0:
        lambda1 = 1.0
    if s0 + s1 < 2.0:
        lambda0 = 1.0
    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = V1.outer_product(U2) - V2.outer_product(U1)
    Q1 = V2.outer_product(U0) - V0.outer_product(U2)
    Q2 = V1.outer_product(U0) - V0.outer_product(U1)
    q0 = flatten_matrix(Q0)
    q1 = flatten_matrix(Q1)
    q2 = flatten_matrix(Q2)

    # ARAP
    d2PsidF2_ARAP = - mu * (lambda0 * (q0.outer_product(q0)) + lambda1 * (q1.outer_product(q1)) + lambda2 * (q2.outer_product(q2)))
    for i in ti.static(range(9)):
        d2PsidF2_ARAP[i, i] += 2 * mu

    g3 = compute_vec_dJdF(F)
    H3 = compute_H3(F)
    J = F.determinant()
    d2PsidF2_FCR = d2PsidF2_ARAP + la * g3.outer_product(g3) + la * (J - 1) * H3
    return d2PsidF2_FCR

@ti.func
def double_dot_product(A, B):

    val = A[0, 0] * B[0, 0] + A[0, 1] * B[0, 1] + A[0, 2] * B[0, 2] +\
          A[1, 0] * B[1, 0] + A[1, 1] * B[1, 1] + A[1, 2] * B[1, 2] +\
          A[2, 0] * B[2, 0] + A[2, 1] * B[2, 1] + A[2, 2] * B[2, 2]

    return val
