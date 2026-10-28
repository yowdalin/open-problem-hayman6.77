# F.Holland: Univalent polynomial problem (Hayman #6.77)

Let $\mathcal P_n$ be the polynomials $p(z) = z + a_2 z^2 + \dots + a_n z^n$ that are univalent in the unit disc. The problem asks for

$$M_n(q) = \max_{p \in \mathcal P_n} \int_0^{2\pi} |p(e^{it})|^q \, dt, \qquad 0 < q < \infty .$$

The natural candidates are the Suffridge polynomials. I attempt both computational and theoretical attacks, starting with the case $n = 3$.

## Contents

| File | What it is |
|---|---|
| `plan677.pdf` | Step-by-step plan: proved lemmas, numerical exploration, rigorous proof strategy for $n=3$ |
| `plan677.tex`, `crosscheck.tex` | LaTeX source of the plan |
| `p677.py` | Toolkit: Dieudonné univalence radius, projection to the boundary of $\mathcal P_n$, $J_q$, multi-start maximizer, Suffridge polynomials |
| `crosscheck_p3.py` | Numerical check of an explicit Schur–Cohn description of univalent cubics against the Dieudonné criterion |

## Running

Requires `numpy` and `scipy`.

```
python p677.py           # unit tests and small exploration
python crosscheck_p3.py  # cubic cross-check (about 2 s)
pdflatex plan677.tex && pdflatex plan677.tex
```

## Status

Numerical exploration only: nothing here is a proof of extremality. For $n = 3$, multi-start optimization found the Suffridge polynomial $S_3$ as the maximizer for $q = 1$ and $q = 2$ (preliminary, few starts).
