"""Independent dense/SciPy reference; run before the Julia refactor.
Requires numpy and scipy. Writes boundary_reference.csv alongside this script.
The differentiation is of the finite-window observable with its true steady state.
"""
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.integrate import solve_ivp, quad_vec
from scipy.special import kv, gamma


def matrices(R, K):
    d = len(R)
    Q = -1j*K - R.conj().T@R/2
    L = np.kron(np.eye(d), Q) + np.kron(Q.conj(), np.eye(d)) + np.kron(R.conj(), R)
    identity = np.eye(d).reshape(-1, order='F')
    rho = np.linalg.solve(L + np.outer(identity, identity)/d, identity/d)
    return Q, L, rho.reshape((d, d), order='F')


def source(u):
    if u == 0:
        return 1/np.sqrt(np.pi)
    return 2*abs(u)*kv(.25, u*u)/(2**2.25*np.sqrt(np.pi)*gamma(1.25)*np.sqrt(abs(u)))


def forward(R, K, beta, T):
    Q, L, rho = matrices(R, K)
    shape = R.shape
    def rhs(u, v):
        X = v.reshape(shape)
        return (2*abs(u)*(Q@X + X@Q.conj().T + R@X@R.conj().T)
                + beta*source(u)*(R@X + X@R.conj().T)).reshape(-1)
    a = np.sqrt(T)
    left = solve_ivp(rhs, (-a, 0), rho.reshape(-1), method='DOP853', atol=2e-12, rtol=2e-11, dense_output=True)
    right = solve_ivp(rhs, (0, a), left.y[:, -1], method='DOP853', atol=2e-12, rtol=2e-11, dense_output=True)
    assert left.success and right.success
    def sol(u):
        return (left.sol(u) if u < 0 else right.sol(u)).reshape(shape)
    return np.trace(sol(a)).real, sol


def gradient(R, K, beta, T):
    Q, L, rho = matrices(R, K)
    value, rho_t = forward(R, K, beta, T)
    shape = R.shape
    def rhs(u, v):
        O = v.reshape(shape)
        return (2*abs(u)*(Q.conj().T@O + O@Q + R.conj().T@O@R)
                + beta*source(u)*(R.conj().T@O + O@R)).reshape(-1)
    a = np.sqrt(T)
    left = solve_ivp(rhs, (-a, 0), np.eye(len(R), dtype=complex).reshape(-1), method='DOP853', atol=2e-12, rtol=2e-11, dense_output=True)
    right = solve_ivp(rhs, (0, a), left.y[:, -1], method='DOP853', atol=2e-12, rtol=2e-11, dense_output=True)
    def obs(u):
        return (left.sol(-u) if u > 0 else right.sol(-u)).reshape(shape)
    def integrand(u):
        X, O = rho_t(u), obs(u)
        return 2*abs(u)*(X@R.conj().T@O - X@O@R.conj().T) + beta*source(u)*X@O
    bulk = quad_vec(integrand, -a, 0, epsabs=1e-10)[0] + quad_vec(integrand, 0, a, epsabs=1e-10)[0]
    A = obs(-a)
    identity = np.eye(len(R)).reshape(-1, order='F')
    rv = rho.reshape(-1, order='F')
    b = (A - np.trace(A@rho)*np.eye(len(R))).reshape(-1, order='F')
    # (-L* + |I><rho|) H = A - <A,rho>I; <H,rho>=0.
    H = np.linalg.solve(-L.conj().T + np.outer(identity, rv.conj()), b).reshape(shape, order='F')
    boundary = rho@R.conj().T@H - rho@H@R.conj().T
    return value, bulk, boundary


def retract(R, K, W, t):
    return R+t*W, K + .5j*t*(W.conj().T@R - R.conj().T@W)


def main():
    rng = np.random.default_rng(718)
    Rbase = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    Kbase = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    Kbase = (Kbase+Kbase.conj().T)/2
    W = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    W /= np.linalg.norm(W)
    rows = []
    for scale in (1., .1, .01):
        R, K = np.sqrt(scale)*Rbase, scale*Kbase
        gap = sorted(-np.linalg.eigvals(matrices(R,K)[1]).real)[1]
        for T in (4., 8., 16., 32.):
            start = perf_counter()
            value, M, B = gradient(R, K, .4, T)
            elapsed = perf_counter()-start
            h = 1e-5
            fd = (forward(*retract(R,K,W,h), .4, T)[0]-forward(*retract(R,K,W,-h), .4, T)[0])/(2*h)
            bulk = 2*np.trace(M@W).real
            full = 2*np.trace((M+B)@W).real
            row = dict(scale=scale, gap=gap, T=T, value=value, finite_difference=fd,
                       bulk=bulk, corrected=full, bulk_error=abs(bulk-fd), corrected_error=abs(full-fd), seconds=elapsed)
            rows.append(row)
            print(f'gap={gap:.4g} T={T:2.0f} bulk error={abs(bulk-fd):.3e} corrected={abs(full-fd):.3e}', flush=True)
    with Path(__file__).with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    main()
