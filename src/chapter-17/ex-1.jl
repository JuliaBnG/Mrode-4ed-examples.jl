"""
    ex_17_1() -> NamedTuple

Sections 17.3 and 17.5: Variance components of a sire model by Henderson's Method 3 and by
fitting canonical sums of squares (Mrode & Pocrnic, 2023, Sections 17.3 and 17.5; Henderson,
1953).

# Model Formulation
For the sire model ``y = Xb + Zs + e`` with ``\\mathrm{var}(s) = I \\sigma_s^2`` and
``\\mathrm{var}(e) = I \\sigma_e^2``:
1. **Henderson's Method 3** (Section 17.3) equates reductions in sums of squares to their
   expectations:
   - Fixed-effects reduction: ``R(b) = y' X (X'X)^- X' y = F``.
   - Full-model reduction: ``R(b, s) = \\hat{b}' X' y + \\hat{s}' Z' y = F + S``.
   - Sire sum of squares adjusted for fixed effects: ``S = R(b, s) - R(b)``.
   - Residual sum of squares: ``R = y'y - R(b, s)``.
   With ``S_X = I - X (X'X)^- X'``, the expectations are
   ``\\mathrm{E}(R) = (N - r(X, Z)) \\sigma_e^2`` and
   ``\\mathrm{E}(S) = \\mathrm{tr}(Z' S_X Z) \\sigma_s^2 + (r(X, Z) - r(X)) \\sigma_e^2``,
   giving unbiased estimators:
   ```math
   \\hat{\\sigma}_e^2 = \\frac{R}{N - r(X, Z)}, \\quad
   \\hat{\\sigma}_s^2 = \\frac{S - (r(X, Z) - r(X)) \\hat{\\sigma}_e^2}{\\mathrm{tr}(Z' S_X
       Z)}.
   ```
2. **Canonical Sums of Squares** (Section 17.5) Diagonalizing ``Z' S_X Z`` forms orthogonal
   sire contrasts ``Q's`` with eigenvalues ``w_i`` and independent mean squares
   ``\\mathrm{E}(\\mathrm{MS}_i) = \\sigma_e^2 + w_i \\sigma_s^2``, along with
   ``\\mathrm{E}(\\mathrm{MS}_e) = \\sigma_e^2``. Writing ``\\mathrm{MS} = C\\theta +
   \\epsilon`` with ``\\theta = [\\sigma_e^2;\\ \\sigma_s^2]``:
   - **OLS**: ``\\hat{\\theta} = (C'C)^{-1} C' \\mathrm{MS}``.
   - **Iterated GLS**: ``\\hat{\\theta} = (C' V^{-1} C)^{-1} C' V^{-1} \\mathrm{MS}`` with
     ``V = \\mathrm{diag}\\!\\left(2\\,\\mathrm{E}(\\mathrm{MS}_i)^2 / \\nu_i\\right)``
     updated at each iterate.

# Key Concepts Illustrated
1. **Method 3 written out from projections**: We compute ``R(b)`` and ``R(b, s)`` directly
   using `pinv`, evaluate ``\\mathrm{tr}(Z' S_X Z)``, and verify the reductions and variance
   estimates against `henderson3`.
2. **OLS and iterated GLS written out**: From the canonical sire contrasts
   (`sire_contrasts`), OLS is simply `C \\ ms`, and iterated GLS is a 3-line loop updating
   the diagonal weight matrix ``V^{-1}``; both are checked against `fit_mean_squares`, and
   the estimates of all three methods are compared in a table `est`.

# Notes on Text Discrepancies
In Section 17.5, the book prints 0.143 and 0.079 for the unweighted OLS fit (exact OLS on
the three mean squares gives 0.151 and 0.062), and prints 0.216 and 0.234 for the "estimated
variances" (here the asymptotic sampling variances are 0.051 and 0.062, with standard errors
0.226 and 0.250). The iterated GLS estimates (0.163, 0.047) match the book.

# Returns
A `NamedTuple` with fields:
- `h`: Henderson's Method 3 result (`F`, `S`, `R`, `ZSZ`, `σ²e`, `σ²s`).
- `c`: Canonical sire contrasts (`ss`, eigenvalues `w`, `Q`, residual `R` and `dfR`).
- `θ`: OLS fit ``(\\hat{\\sigma}_e^2, \\hat{\\sigma}_s^2)`` to the canonical mean squares.
- `θw`: Iterated GLS fit ``(\\hat{\\sigma}_e^2, \\hat{\\sigma}_s^2)``.
- `est`: Table comparing the estimates of the three methods.
"""
function ex_17_1()
    println("Sections 17.3 and 17.5: variance components of a sire model")

    # Step 1: Records (one row per progeny) and design matrices
    rec = DataFrame(sire = [2, 1, 3, 2], y = [2.9, 4.0, 3.5, 3.5])
    y = rec.y
    X = ones(nrow(rec), 1)
    Z = Matrix(first(incidence(rec.sire)))

    # Step 2: Henderson's Method 3, written out from the reductions
    #    R(b) = y'X(X'X)⁻X'y and R(b, s) with [X Z], so S = R(b, s) - R(b) and
    #    R = y'y - R(b, s); E(S) = tr(Z'S_X Z)σ²s + (r(X, Z) - r(X))σ²e
    h = henderson3(y, X, Z)
    red(T) = y' * T * pinv(T'T) * T' * y
    W = [X Z]
    Rb, Rbs = red(X), red(W)
    Sx = I - X * pinv(X'X) * X'
    rX, rXZ = rank(X), rank(W)
    σ²e3 = (y'y - Rbs) / (length(y) - rXZ)
    σ²s3 = (Rbs - Rb - (rXZ - rX) * σ²e3) / tr(Z' * Sx * Z)
    check(
        "F, S, R and σ²e, σ²s = henderson3()",
        [Rb, Rbs - Rb, y'y - Rbs, σ²e3, σ²s3],
        [h.F, h.S, h.R, h.σ²e, h.σ²s];
        atol = 1e-12,
    )
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

    #    written out: OLS is C \ ms (least squares for a tall C); GLS iterates
    #    θ = (C'V⁻¹C)⁻¹C'V⁻¹ms with V = diag(2E(msᵢ)²/νᵢ) at the current θ
    θe = C \ ms
    for _ = 1:100
        Vi = Diagonal(df ./ (2 .* (C * θe) .^ 2))
        θe = (C' * Vi * C) \ (C' * Vi * ms)
    end
    check(
        "OLS = C \\ ms, GLS iteration (fit_mean_squares)",
        [C \ ms; θe],
        [θ; θw];
        atol = 1e-8,
    )

    est = DataFrame(
        method = ["Henderson 3", "OLS", "iterated GLS"],
        σ²e = [h.σ²e, θ[1], θw[1]],
        σ²s = [h.σ²s, θ[2], θw[2]],
    )
    show_table(est)

    # The book's "estimated variances" 0.216 and 0.234 are not reproduced;
    # the sampling variances here are (0.051, 0.062), SEs (0.226, 0.250).
    @printf(
        "  sampling variances %.3f, %.3f; standard errors %.3f, %.3f\n",
        diag(V)...,
        sqrt.(diag(V))...
    )
    (; h, c, θ, θw, est)
end
