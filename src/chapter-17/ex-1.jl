"""
    ex_17_1() -> NamedTuple

Run Sections 17.3 and 17.5 from Mrode & Pocrnic (2023): Estimation of variance components
in a sire model using Henderson's Method 3 and canonical sums of squares
(Ordinary Least Squares and iterated Generalized Least Squares).

# Henderson's Method 3 (Section 17.3)
For the sire model:
```math
\\mathbf{y} = X\\mathbf{b} + Z\\mathbf{s} + \\mathbf{e}
```
with ``\\mathrm{Var}(\\mathbf{s}) = I \\sigma_s^2`` and
``\\mathrm{Var}(\\mathbf{e}) = I \\sigma_e^2``. Reductions in sums of squares are computed
as:
- Fixed-effects reduction: ``R(\\mathbf{b}) = \\mathbf{y}' X (X'X)^- X' \\mathbf{y} = F``
-  Full-model reduction:
  ``R(\\mathbf{b}, \\mathbf{s}) = \\hat{\\mathbf{b}}' X' \\mathbf{y} + \\hat{\\mathbf{s}}'
  Z' \\mathbf{y} = F + S``
-  Sire sum of squares adjusted for fixed effects:
  ``R(\\mathbf{s} \\mid \\mathbf{b}) = R(\\mathbf{b}, \\mathbf{s}) - R(\\mathbf{b}) = S``
-  Residual sum of squares:
  ``\\mathrm{SSE} = \\mathbf{y}' \\mathbf{y} - R(\\mathbf{b}, \\mathbf{s}) = R``

Equating quadratic forms to their expectations:
```math
\\begin{aligned}
\\mathbb{E}[R] &= (N - r(X, Z)) \\sigma_e^2 \\\\
\\mathbb{E}[S] &= \\mathrm{tr}(Z' S_X Z) \\sigma_s^2 + (r(X, Z) - r(X)) \\sigma_e^2
\\end{aligned}
```
where ``S_X = I - X (X'X)^- X'``. Unbiased estimates are:
```math
\\hat{\\sigma}_e^2 = \\frac{R}{N - r(X, Z)}, \\quad
\\hat{\\sigma}_s^2 = \\frac{S - (r(X, Z) - r(X)) \\hat{\\sigma}_e^2}{\\mathrm{tr}(Z' S_X Z)}
```

# Canonical Sums of Squares & GLS (Section 17.5)
Constructs mutually orthogonal sire contrasts ``Q' \\mathbf{s}`` with eigenvalues ``w_i``:
```math
\\mathbb{E}[\\mathrm{MS}_i] = \\sigma_e^2 + w_i \\sigma_s^2, \\quad
    \\mathbb{E}[\\mathrm{MS}_e] = \\sigma_e^2
```
Written in regression form
``\\mathbf{MS} = C \\boldsymbol{\\theta} + \\boldsymbol{\\epsilon}`` with
``\\boldsymbol{\\theta} = [\\sigma_e^2 \\; \\sigma_s^2]'``:
- **OLS**: ``\\hat{\\boldsymbol{\\theta}} = (C'C)^{-1} C' \\mathbf{MS}``
-  **Iterated GLS**:
  ``\\hat{\\boldsymbol{\\theta}} = (C' V^{-1} C)^{-1} C' V^{-1} \\mathbf{MS}`` where
  ``V = 2 \\,\\mathrm{diag}\\left(\\frac{(\\mathbb{E}[\\mathrm{MS}_i])^2}{\\nu_i}\\right)``,
  iterated until convergence.

# Returns
A `NamedTuple` with fields:
- `h`: Henderson's method 3 (reductions, degrees of freedom and estimates).
- `c`: Canonical sire contrasts (sums of squares, eigenvalues ``w_i``, residual).
- `θ`: OLS fit ``(\\hat\\sigma_e^2, \\hat\\sigma_s^2)`` to the mean squares.
- `θw`: Iterated GLS fit ``(\\hat\\sigma_e^2, \\hat\\sigma_s^2)``.
"""
function ex_17_1()
    println("Sections 17.3 and 17.5: variance components of a sire model")

    # Step 1: Data and design matrices
    sire = [2, 1, 3, 2]
    y = [2.9, 4.0, 3.5, 3.5]
    X = ones(4, 1)
    Z = Matrix(first(incidence(sire)))

    # Step 2: Henderson's Method 3
    h = henderson3(y, X, Z)
    check("F, S, R", [h.F, h.S, h.R], [48.3025, 0.4275, 0.1800]; atol = 1e-10)
    check("Z'SZ", h.ZSZ, [0.75 -0.5 -0.25; -0.5 1 -0.5; -0.25 -0.5 0.75]; atol = 1e-12)
    check("σ²e, σ²s (method 3)", [h.σ²e, h.σ²s], [0.18, 0.027])

    # Step 3: Canonical sire contrasts
    c = sire_contrasts(y, X, Z, Matrix(1.0I, 3, 3))
    check("W", c.w, [1.5, 1.0]; atol = 1e-12)
    check("|Q|", abs.(c.Q'), [0.3333 0.6667 0.3333; 0.7071 0 0.7071]; atol = 1e-4)
    check("contrast sums of squares", c.ss, [0.3025, 0.1250]; atol = 1e-12)

    # Step 4: Fitting mean squares by OLS and iterated GLS
    ms = [c.ss; c.R / c.dfR]
    df = [1, 1, c.dfR]
    C = [ones(3) [c.w; 0]]

    # The book reports 0.143 and 0.079 for the unweighted fit; ordinary
    # least squares on these three sums of squares gives 0.151 and 0.062.
    θ, _ = fit_mean_squares(ms, df, C; weighted = false)
    @printf("  OLS fit: σ²e = %.3f, σ²s = %.3f (book 0.143, 0.079)\n", θ...)
    θw, V = fit_mean_squares(ms, df, C)
    check("iterated GLS fit: σ²e, σ²s", θw, [0.163, 0.047])

    # The book's "estimated variances" 0.216 and 0.234 are not reproduced;
    # the sampling variances here are (0.051, 0.062), SEs (0.226, 0.250).
    @printf(
        "  sampling variances %.3f, %.3f; standard errors %.3f, %.3f\n",
        diag(V)...,
        sqrt.(diag(V))...
    )
    (; h, c, θ, θw)
end
