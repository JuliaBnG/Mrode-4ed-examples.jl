"""
    ex_17_2() -> NamedTuple

Run Section 17.7 from Mrode & Pocrnic (2023): Restricted Maximum Likelihood (REML)
estimation of variance components in an animal model using:
1. Average Information REML (AI-REML, Table 17.3)
2. Expectation-Maximization REML (EM-REML, Eqns 17.5–17.6)

# Model & Likelihood
For the animal model:
```math
\\mathbf{y} = X\\mathbf{b} + Z\\mathbf{a} + \\mathbf{e}
```
with ``\\mathbf{a} \\sim N(0, A \\sigma_a^2)``, ``\\mathbf{e} \\sim N(0, I \\sigma_e^2)``,
and ``V = ZAZ' \\sigma_a^2 + I \\sigma_e^2``. The restricted log-likelihood is:
```math
\\log L_R = -\\frac{1}{2} \\left( \\log |C| + \\log |A| + N \\log \\sigma_e^2 + q \\log
    \\sigma_a^2 + \\mathbf{y}' \\mathbf{P} \\mathbf{y} \\right)
```
up to a constant, where ``C`` is the coefficient matrix of the MME built with
``R^{-1} = I/\\sigma_e^2`` and ``A^{-1}/\\sigma_a^2``, ``q`` is the number of animals and
``\\mathbf{P} = V^{-1} - V^{-1} X (X' V^{-1} X)^- X' V^{-1}``.

# Algorithms
-  **AI-REML**: Averages the observed and expected information matrices, eliminating
  expensive trace calculations of matrix inverses by using data-driven quadratic forms:
  ```math
  \\boldsymbol{\\theta}^{(k+1)} = \\boldsymbol{\\theta}^{(k)} +
      \\mathcal{I}_{\\mathrm{AI}}^{-1} \\nabla \\log L_R
  ```
  The inverse Average Information matrix ``\\mathcal{I}_{\\mathrm{AI}}^{-1}`` at convergence
  provides the asymptotic sampling covariance matrix for
  ``(\\hat{\\sigma}_e^2, \\hat{\\sigma}_a^2)``.
- **EM-REML**: Iterative updates based on conditional expectations of random effects:
  ```math
  \\sigma_a^{2(k+1)} = \\frac{\\hat{\\mathbf{a}}' A^{-1} \\hat{\\mathbf{a}} +
      \\mathrm{tr}(A^{-1} C^{aa}) \\sigma_e^{2(k)}}{q}
  ```
  Guaranteed monotonic convergence in likelihood, though slower than AI-REML.

# Returns
A `NamedTuple` with fields:
- `r`: AI-REML result (`REMLResult`: estimates, log-likelihood, inverse AI matrix,
  iteration history and final MME solutions).
- `em`: EM-REML result (`REMLResult`).
"""
function ex_17_2()
    println("Section 17.7: REML for an animal model")

    # Step 1: Pedigree and phenotype data
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    y = [2.6, 0.1, 1.0, 3.0, 1.0]
    X, _ = incidence(sex; levels = ["M", "F"])
    a = RandomEffect(incidence(calf, 8), ainv(ped), 0.2)

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
        y - [X a.Z] * x,
        [0.2022, -0.3661, 0.3661, 0.6374, -0.8395];
        atol = 1e-4,
    )
    check(
        "C²²σ²e, first row",
        lhs_inverse(m)[3, 3:10] .* 0.4,
        [0.1884, 0.0028, 0.0131, 0.0878, 0.0180, 0.0883, 0.0554, 0.0537];
        atol = 1e-4,
    )

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
    em = reml(y, X, [a]; σ²e = 0.4, method = :em, maxiter = 1000)
    check(
        "EM after 1000 iterates (σ²e, σ²a, logL)",
        [em.history[:, end]; em.logLs[end]],
        [0.4842, 0.5504, -2.1817],
    )
    (; r, em)
end
