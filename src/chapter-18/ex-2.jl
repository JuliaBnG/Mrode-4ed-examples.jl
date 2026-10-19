"""
    ex_18_2() -> NamedTuple

Example 18.2: Multivariate block Gibbs sampling for a bivariate animal model (Mrode &
Pocrnic, 2023, Section 18.3).

# Model Formulation
For ``t`` traits measured on ``N`` animals with observation matrix ``Y`` (``N \\times t``),
genetic covariance ``\\mathrm{var}(u) = G_0 \\otimes A``, and residual covariance
``\\mathrm{var}(e) = R_0 \\otimes I``:
- **Inverse-Wishart conjugate priors**: Placing ``G_0 \\sim W^{-1}(\\nu_u, V_u)`` and
  ``R_0 \\sim W^{-1}(\\nu_e, V_e)`` yields inverse-Wishart full conditional distributions:
  ```math
  \\begin{aligned}
  G_0 \\mid u &\\sim W^{-1}\\!\\left(\\nu_u + q, \\; V_u + U' A^{-1} U\\right), \\\\
  R_0 \\mid e &\\sim W^{-1}\\!\\left(\\nu_e + N, \\; V_e + E' E\\right),
  \\end{aligned}
  ```
  where ``U`` (``q \\times t``) and ``E`` (``N \\times t``) are the matrices of animal
  breeding values and residuals across traits.
- **Data augmentation for missing records**: When an animal has trait 1 (WWG) observed but
  trait 2 (PWG) missing, the missing phenotype is sampled at each Gibbs sweep from its
  conditional normal distribution given the observed trait:
  ```math
  y_{i,2} \\mid y_{i,1}, b, u, R_0 \\sim N\\!\\left(
  x_i'b_2 + u_{i,2} + \\frac{r_{12}}{r_{11}}(y_{i,1} - x_i'b_1 - u_{i,1}), \\;
  r_{22} - \\frac{r_{12}^2}{r_{11}}\\right),
  ```
  restoring a complete ``N \\times t`` data matrix so that the complete-data MME and residual
  sum-of-squares updates can be used directly.

# Key Concepts Illustrated
1. **Equivalence with multi-trait BLUP**: Using the 8-animal data table of [`ex_06_1`](@ref),
   we run 8 independent chains with ``R_0`` and ``G_0`` fixed (`gibbs_mt`) and verify that
   the posterior means converge to the exact bivariate BLUP solutions (stored side by side in
   `ped`).
2. **Data augmentation in action**: Setting calf 4's PWG record to `missing`, we compute the
   conditional normal distribution ``N(\\mu_{2\\mid 1}, \\sigma_{e_{2\\mid 1}}^2)`` explicitly
   at the BLUP solutions and verify that the augmented Gibbs sampler converges to the
   BLUP of the same data with calf 4's PWG missing (the missing-record pattern of
   [`ex_06_2`](@ref)).
3. **Sampling ``R_0`` and ``G_0``**: With proper inverse-Wishart priors (``\\nu = 10``)
   centred on ``R_0`` and ``G_0``, we compare prior and posterior covariance means in a
   summary table.

# Returns
A `NamedTuple` with fields:
- `f`: Full multivariate Gibbs chain with ``R_0`` and ``G_0`` sampled (`GibbsResult`).
- `ped`: The data table with `blup_wwg`, `blup_pwg`, `post_wwg`, and `post_pwg` added.
"""
function ex_18_2()
    println("Example 18.2: multivariate Gibbs sampling")

    # Step 1: Pedigree and both traits of Example 6.1, one row per animal
    no3 = fill(missing, 3)
    ped = DataFrame(
        calf = 1:8,
        sex = [no3; ["M", "F", "F", "M", "M"]],
        sire = [0, 0, 0, 1, 3, 1, 4, 3],
        dam = [0, 0, 0, 0, 2, 2, 5, 6],
        wwg = [no3; [4.5, 2.9, 3.9, 3.5, 5.0]],
        pwg = [no3; [6.8, 5.0, 6.8, 6.0, 7.5]],
    )
    rec = dropmissing(ped, [:wwg, :pwg])
    Y = Matrix(rec[:, [:wwg, :pwg]])
    G0 = [20.0 18; 18 40]
    R0 = [40.0 11; 11 30]
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Z = incidence(rec.calf, nrow(ped))
    Ai = ainv(ped)

    # Step 2: Multi-trait BLUP of Example 6.1 for analytical reference
    y, obs = mt_stack(Y)
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [X, X])),
        [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0)];
        Rinv = mt_rinv(R0, obs),
    )
    b, (u,) = solutions(m, solve_mme(m))

    # Step 3: Fixed (co)variances: verify posterior means converge to BLUP (8 independent chains)
    chains(Yd, seed) = reduce(
        hcat,
        [
            gibbs_mt(
                Yd,
                X,
                Z,
                Ai;
                R0,
                G0,
                estimate_variances = false,
                niter = 50_000,
                burnin = 1_000,
                rng = Xoshiro(seed + c),
            ).x for c = 1:8
        ],
    )
    pm = chains(Y, 100)
    mc_check("posterior means vs BLUP", pm, vec([reshape(b, 2, 2); u]))
    ped.blup_wwg, ped.blup_pwg = u[:, 1], u[:, 2]
    pu = reshape(vec(mean(pm; dims = 2)), :, 2)[3:end, :]
    ped.post_wwg, ped.post_pwg = pu[:, 1], pu[:, 2]

    # Step 4: Missing PWG record (Example 6.2 pattern) handled via data augmentation
    Ym = Matrix{Union{Missing,Float64}}(Y)
    Ym[1, 2] = missing
    y2, obs2 = mt_stack(Ym)
    m2 = mme(
        y2,
        reduce(hcat, mt_blocks(obs2, [X, X])),
        [RandomEffect(mt_blocks(obs2, [Z, Z]), Ai, G0)];
        Rinv = mt_rinv(R0, obs2),
    )
    b2, (u2,) = solutions(m2, solve_mme(m2))
    #    In every round the missing PWG of calf 4 is drawn from its distribution
    #    given its WWG record: mean x'b₂ + u₂ + (r₁₂/r₁₁)(y₁ - x'b₁ - u₁), variance
    #    r₂₂ - r₁₂²/r₁₁; at the BLUP solutions:
    B2 = reshape(b2, 2, 2)
    η = X[1, :]' * B2 .+ u2[4, :]'               # expected WWG, PWG of calf 4
    μ21 = η[2] + R0[1, 2] / R0[1, 1] * (Y[1, 1] - η[1])
    @printf(
        "  calf 4, missing PWG: N(%.3f, %.3f) given WWG = %.1f\n",
        μ21,
        R0[2, 2] - R0[1, 2]^2 / R0[1, 1],
        Y[1, 1]
    )
    mc_check(
        "missing record: posterior means vs BLUP",
        chains(Ym, 200),
        vec([reshape(b2, 2, 2); u2]),
    )

    # Step 5: (Co)variances sampled with proper inverse-Wishart priors with means R0, G0
    #         (V/(ν - t - 1) with V = R0(ν - 3), t = 2 traits)
    ν = 10
    f = gibbs_mt(
        Y,
        X,
        Z,
        Ai;
        R0,
        G0,
        νe = ν,
        Ve = R0 * (ν - 3),
        νu = ν,
        Vu = G0 * (ν - 3),
        niter = 50_000,
        burnin = 5_000,
        rng = Xoshiro(5),
    )
    show_table(
        parameter = ["r11", "r21", "r22", "g11", "g21", "g22"],
        prior_mean = [R0[1, 1], R0[2, 1], R0[2, 2], G0[1, 1], G0[2, 1], G0[2, 2]],
        posterior_mean = vec(mean(f.samples; dims = 2)),
    )
    println("\nBLUP and posterior means (variances fixed) of both traits:")
    show_table(ped)
    (; f, ped)
end
