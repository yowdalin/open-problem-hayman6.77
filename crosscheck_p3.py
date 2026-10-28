"""
crosscheck_p3.py -- compare two descriptions of univalent cubics z + a2 z^2 + a3 z^3.

(A) Schur-Cohn: for all x = cos(theta) in (0,1), with b = 2 a2 x, c = a3 (4x^2 - 1):
        |c| < 1   and   |b - conj(b) c| < 1 - |c|^2
(B) Dieudonne radius R(p) > 1 (computed by root-finding on a theta grid).

Both say "strictly inside P_3". Points with |R - 1| < tol are skipped (boundary is
where the open/closed distinction and grid error matter). NUMERICAL ONLY.
"""
import numpy as np

def schur_cohn_inside(a2, a3, nx=3000):
    x = np.linspace(1e-4, 1 - 1e-4, nx)
    b = 2 * a2 * x
    c = a3 * (4 * x**2 - 1)
    return bool(np.all((np.abs(c) < 1) & (np.abs(b - np.conj(b) * c) < 1 - np.abs(c) ** 2)))

def dieudonne_radius_cubic(a2, a3, nt=3000):
    th = np.linspace(1e-6, np.pi / 2, nt)
    x = np.cos(th)
    b = 2 * a2 * x                 # D(z) = 1 + b z + c z^2
    c = a3 * (4 * x**2 - 1)
    # roots of c z^2 + b z + 1 via the quadratic formula, vectorized over theta
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = np.sqrt(b.astype(complex) ** 2 - 4 * c)
        r1 = np.where(np.abs(c) > 1e-14, (-b + disc) / (2 * c), -1 / b)
        r2 = np.where(np.abs(c) > 1e-14, (-b - disc) / (2 * c), np.inf)
    R = np.nanmin(np.minimum(np.abs(r1), np.abs(r2)))
    return R

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N, tol = 3000, 1e-3
    agree = disagree = skipped = 0
    bad = []
    for _ in range(N):
        # sample around the plausible region: |a2| <= 1.2, |a3| <= 0.4
        a2 = 1.2 * np.sqrt(rng.random()) * np.exp(2j * np.pi * rng.random())
        a3 = 0.4 * np.sqrt(rng.random()) * np.exp(2j * np.pi * rng.random())
        R = dieudonne_radius_cubic(a2, a3)
        if abs(R - 1) < tol:
            skipped += 1
            continue
        if schur_cohn_inside(a2, a3) == (R > 1):
            agree += 1
        else:
            disagree += 1
            bad.append((a2, a3, R))
    print(f"agree {agree}  disagree {disagree}  skipped(|R-1|<{tol}) {skipped}")
    for b_ in bad[:10]:
        print("  mismatch:", b_)
