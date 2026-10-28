"""
p677.py -- starter toolkit for Hayman-Lingham Problem 6.77
Maximize J_q(p) = int_0^{2pi} |p(e^{it})|^q dt over univalent p(z) = z + a_2 z^2 + ... + a_n z^n.

Everything here is NUMERICAL (floating point). It is for exploration only.
Rigorous certificates (interval arithmetic) come later, see the plan PDF.

Key facts used (proved in the plan):
 * Dieudonne criterion: p univalent in D  <=>  D_theta(z) = sum_k a_k sin(k theta)/sin(theta) z^{k-1}
   has no zeros in |z|<1 for every 0 < theta < pi/2.
 * Dilation: p_r(z) = p(rz)/r has coefficients a_k r^{k-1}, and D_theta for p_r is D_theta(rz).
   Hence  R(p) := inf_theta min_j |root_j(D_theta)|  is the largest r with p_r univalent in D,
   and p_{R(p)} lies exactly on the boundary of the univalent class.
 * J_q is plurisubharmonic in the coefficients, so maxima lie on the boundary.
"""
import numpy as np

def suffridge(n):
    """Coefficients [a_1,...,a_n] of the Suffridge polynomial S_n (a_1 = 1)."""
    k = np.arange(1, n + 1)
    return ((n - k + 1) / n) * np.sin(k * np.pi / (n + 1)) / np.sin(np.pi / (n + 1))

def dieudonne_radius(a, n_theta=2000):
    """a = [a_1=1, a_2, ..., a_n] (complex). Returns R(p) = inf over theta in (0, pi/2]
    of the smallest root modulus of D_theta. p is univalent in D iff R(p) >= 1 (numerically)."""
    a = np.asarray(a, dtype=complex); n = len(a)
    k = np.arange(1, n + 1)
    # theta grid incl. points near 0 (limit gives p'(z)) and pi/2
    th = np.concatenate([np.linspace(1e-6, np.pi / 2, n_theta)])
    R = np.inf
    for t in th:
        c = a * np.sin(k * t) / np.sin(t)          # coefficient of z^{k-1}
        coeffs = c[::-1]                            # highest degree first
        # strip numerically-zero leading coefficients (degree drop => root at infinity)
        while len(coeffs) > 1 and abs(coeffs[0]) < 1e-14:
            coeffs = coeffs[1:]
        if len(coeffs) <= 1:
            continue
        R = min(R, np.min(np.abs(np.roots(coeffs))))
    return R

def project_to_boundary(a):
    """Dilate p so that it lies exactly on the boundary of the univalent class."""
    a = np.asarray(a, dtype=complex); R = dieudonne_radius(a)
    k = np.arange(len(a))                            # exponent k-1 for a_k
    return a * R**k, R

def J(a, q, N=4096):
    """J_q(p) = int_0^{2pi} |p(e^{it})|^q dt by the trapezoid rule (spectrally accurate:
    the integrand is smooth because a univalent p with p(0)=0 never vanishes on |z|=1)."""
    a = np.asarray(a, dtype=complex)
    t = 2 * np.pi * np.arange(N) / N; z = np.exp(1j * t)
    p = np.polyval(np.r_[a[::-1], 0], z)            # p(z) = sum a_k z^k
    return 2 * np.pi * np.mean(np.abs(p) ** q)

def objective(x, n, q):
    """x = real vector of length 2(n-1): (Re a_2, Im a_2, ..., Re a_n, Im a_n).
    Returns -J_q of the boundary-projected polynomial (for minimizers)."""
    a = np.r_[1.0, x[0::2] + 1j * x[1::2]]
    b, _ = project_to_boundary(a)
    return -J(b, q)

def maximize(n, q, starts=20, seed=0, n_theta=400):
    """Multi-start Nelder-Mead over boundary polynomials. Returns best value and coefficients."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    global dieudonne_radius
    best = (np.inf, None)
    for s in range(starts):
        x0 = rng.normal(0, 0.5, 2 * (n - 1))
        res = minimize(objective, x0, args=(n, q), method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-10})
        if res.fun < best[0]:
            best = (res.fun, res.x)
    x = best[1]; a = np.r_[1.0, x[0::2] + 1j * x[1::2]]
    b, _ = project_to_boundary(a)
    # normalize rotation: make a_2 real and >= 0 (p -> e^{-i phi} p(e^{i phi} z) multiplies a_k by e^{i(k-1)phi})
    if abs(b[1]) > 1e-12:
        phi = -np.angle(b[1]); b = b * np.exp(1j * phi * np.arange(len(b)))
    return -best[0], b

if __name__ == "__main__":
    # ---- unit tests ----
    for a2, expect in [(0.5, True), (0.49, True), (0.51, False)]:
        R = dieudonne_radius([1, a2])
        print(f"z + {a2} z^2 : R = {R:.6f}  univalent={R >= 1 - 1e-9}  (expected {expect})")
    for n in [2, 3, 4, 5]:
        print(f"Suffridge S_{n}: R = {dieudonne_radius(suffridge(n)):.6f} (expected 1: on the boundary)")
    print("Koebe truncation z+2z^2+3z^3: R =", round(dieudonne_radius([1, 2, 3]), 4), "(expected < 1)")
