"""
    ex_17_2() -> NamedTuple

Section 17.7: Restricted Maximum Likelihood (REML) for an animal model by Average
Information (AI-REML) and Expectation–Maximization (EM-REML) (Mrode & Pocrnic, 2023,
Section 17.7; Patterson & Thompson, 1971; Gilmour et al., 1995).

# Model Formulation
For the univariate animal model ``y = Xb + Za + e`` with ``a \\sim N(0, A \\sigma_a^2)``,
``e \\sim N(0, I \\sigma_e^2)``, and phenotypic covariance ``V = ZAZ' \\sigma_a^2 +
I \\sigma_e^2``, the restricted log-likelihood (up to an additive constant) is:
```math
\\log L_R = -\\frac{1}{2} \\left( \\log |C| + \\log |A| + N \\log \\sigma_e^2 + q \\log
    \\sigma_a^2 + y' P y \\right)
```
where ``C`` is the MME coefficient matrix in the general ``R^{-1}`` form (built with
``R^{-1} = I/\\sigma_e^2`` and ``A^{-1}/\\sigma_a^2``), ``q`` is the number of animals in
the pedigree, and ``y'Py = y'R^{-1}y - \\hat{\\theta}' r`` is the generalized residual
quadratic form.

# Key Concepts Illustrated
1. **Restricted log-likelihood and first EM iterate written out from the MME**: Using the
   8-animal pedigree and 5 records in `ped`, we assemble the MME at the starting values
   ``(\\sigma_e^2 = 0.4,\\, \\sigma_a^2 = 0.2)`` and write out both ``\\log L_R`` and the
   first EM-REML update (Eqns 17.5–17.6):
   ```math
   \\sigma_a^{2(1)} = \\frac{\\hat{a}' A^{-1} \\hat{a} + \\mathrm{tr}(A^{-1} C^{aa})
   \\sigma_e^2}{q}, \\quad
   \\sigma_e^{2(1)} = \\frac{y'y - \\hat{\\theta}' W'y}{N - r(X)},
   ```
   verifying both against `reml`.
2. **AI-REML vs EM-REML convergence**: AI-REML replaces expensive trace evaluations with
   working-variate quadratic forms (the Average Information matrix
   ``\\mathcal{I}_{\\mathrm{AI}}``) and converges in 5 iterates, whereas EM-REML is
   first-order and requires ~1000 iterates to reach the same maximum. At convergence,
   ``\\mathcal{I}_{\\mathrm{AI}}^{-1}`` provides the asymptotic sampling covariance matrix
   of ``(\\hat{\\sigma}_e^2, \\hat{\\sigma}_a^2)``, and the final EBVs are added as a column
   of `ped`.

# Note on Text Discrepancy
At the starting values, the book prints calf 4's breeding value as ``\\hat{a}_4 = -0.254``;
its sign is ``+0.254``, which is required to reproduce the book's own residual for calf 4
(``2.6 - 2.144 - 0.254 = 0.2022``).

# Returns
A `NamedTuple` with fields:
- `r`: AI-REML result (`REMLResult`: estimates, log-likelihoods, inverse AI matrix,
  iteration history, and final MME solutions).
- `em`: EM-REML result (`REMLResult`).
- `ped`: The data table with the EBVs at the AI-REML estimates (`ebv`) added.
"""
function ex_17_2()
    println("Section 17.7: REML for an animal model")

    # Step 1: Pedigree and records, one row per animal (IDs are row numbers;
    #    founders 1–3 have no records)
    no3 = fill(missing, 3)
    ped = DataFrame(
        calf = 1:8,
        sex = [no3; ["M", "F", "F", "M", "M"]],
        sire = [0, 0, 0, 1, 3, 1, 4, 3],
        dam = [0, 0, 0, 0, 2, 2, 5, 6],
        y = [no3; [2.6, 0.1, 1.0, 3.0, 1.0]],
    )
    rec = dropmissing(ped, :y)
    y = rec.y
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Z = incidence(rec.calf, nrow(ped))
    Ai = ainv(ped)
    a = RandomEffect(Z, Ai, 0.2)

    # Step 2: First iterate in detail (starting values: σ²e = 0.4, σ²a = 0.2)
    m = mme(y, X, [a]; σ²e = 0.4)
    x = solve_mme(m)
    # The book prints â₄ = −0.254; its own residual of calf 4 (0.2022) needs +0.254
    check(
        "MME solutions at σ²e = 0.4, σ²a = 0.2",
        x,
        [2.144, 0.602, 0.117, -0.025, -0.222, 0.254, -0.135, 0.032, 0.219, -0.305],
    )
    check(
        "residuals",
        y - [X Z] * x,
        [0.2022, -0.3661, 0.3661, 0.6374, -0.8395];
        atol = 1e-4,
    )
    check(
        "C²²σ²e, first row",
        lhs_inverse(m)[3, 3:10] .* 0.4,
        [0.1884, 0.0028, 0.0131, 0.0878, 0.0180, 0.0883, 0.0554, 0.0537];
        atol = 1e-4,
    )

    #    Restricted log-likelihood at the starting values, written out: with C and
    #    r the MME in the R⁻¹ form, y'Py = y'R⁻¹y - θ̂'r and
    #    log L_R = -½(log|C| + log|A| + N log σ²e + q log σ²a + y'Py)
    σ²e0, σ²a0 = 0.4, 0.2
    N, q = length(y), nrow(ped)
    W = [X Z]
    C = Matrix([X'X X'Z; Z'X Z'Z + σ²e0 / σ²a0 * Ai]) / σ²e0
    rh = W'y / σ²e0
    θ = C \ rh
    logL0 =
        -(
            logdet(C) +
            logdet(inv(Matrix(Ai))) +
            N * log(σ²e0) +
            q * log(σ²a0) +
            y'y / σ²e0 - θ'rh
        ) / 2

    # Step 3: AI-REML estimation
    r = reml(y, X, [a]; σ²e = 0.4, method = :ai)
    show_table(
        iterate = 1:length(r.logLs),
        σ²e = r.history[1, 1:(end-1)],
        σ²a = r.history[2, 1:(end-1)],
        logL = r.logLs;
        digits = 4,
    )
    check(
        "AI-REML iterates σ²e",
        r.history[1, 1:6],
        [0.4000, 0.4838, 0.4910, 0.4839, 0.4835, 0.4835];
        atol = 1e-4,
    )
    check(
        "AI-REML iterates σ²a",
        r.history[2, 1:6],
        [0.2000, 0.3695, 0.5126, 0.5500, 0.5514, 0.5514];
        atol = 1e-4,
    )
    check("log L_R at the starting values, written out", logL0, r.logLs[1]; atol = 1e-10)
    check(
        "log-likelihoods",
        r.logLs[1:6],
        [-2.3852, -2.2021, -2.1821, -2.1817, -2.1817, -2.1817];
        atol = 1e-4,
    )
    check("AI⁻¹ at convergence", r.AIinv, [2.4436 -3.2532; -3.2532 5.3481]; atol = 1e-3)
    @printf("  standard errors: σ²e %.3f, σ²a %.3f\n", sqrt.(diag(r.AIinv))...)

    # Step 4: EM-REML comparison
    e1 = reml(y, X, [a]; σ²e = 0.4, method = :em, maxiter = 1)
    check("first EM iterate (σ²e, σ²a)", e1.history[:, 2], [0.6426, 0.2125])
    #    written out (Eqns 17.5–17.6), from the MME at the starting values:
    #    σ²a = (â'A⁻¹â + tr(A⁻¹Cᵃᵃ)σ²e)/q,  σ²e = (y'y - θ̂'W'y)/(N - r(X))
    Ci = inv(C * σ²e0)                          # inverse of the λ-form coefficient matrix
    â = θ[3:end]
    σ²a1 = (â' * Ai * â + tr(Ai * Ci[3:end, 3:end]) * σ²e0) / q
    σ²e1 = (y'y - θ' * (W'y)) / (N - rank(X))
    check("first EM iterate, written out", [σ²e1, σ²a1], e1.history[:, 2]; atol = 1e-10)
    em = reml(y, X, [a]; σ²e = 0.4, method = :em, maxiter = 1000)
    check(
        "EM after 1000 iterates (σ²e, σ²a, logL)",
        [em.history[:, end]; em.logLs[end]],
        [0.4842, 0.5504, -2.1817],
    )
    ped.ebv = r.solutions[3:end]
    println("\nEBV at the AI-REML estimates:")
    show_table(ped)

    (; r, em, ped)
end
