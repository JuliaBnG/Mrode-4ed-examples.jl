"""
    ex_18_2() -> NamedTuple

Run Example 18.2 from Mrode & Pocrnic (2023): Multivariate block Gibbs sampling
for a multi-trait animal model (WWG and PWG of Example 6.1).

# Model & Priors
For ``t`` traits with phenotypic observations ``Y`` (``N \\times t``):
```math
\\mathrm{vec}(Y') = (X \\otimes I) \\mathbf{b} + (Z \\otimes I) \\mathbf{u} + \\mathbf{e}
```
with genetic covariances ``\\mathbf{u} \\sim N(0, G_0 \\otimes A)`` and
residuals ``\\mathbf{e} \\sim N(0, R_0 \\otimes I)``.

- **Inverse-Wishart Conjugate Priors**:
  ```math
  G_0 \\sim W^{-1}(\\nu_u, V_u), \\quad R_0 \\sim W^{-1}(\\nu_e, V_e)
  ```
  with conditional posteriors:
  ```math
  \\begin{aligned}
  G_0 \\mid \\mathbf{u} &\\sim W^{-1}\\left(\\nu_u + q, \\; V_u + \\sum_{i,j} A^{ij}
      \\mathbf{u}_i \\mathbf{u}_j'\\right) \\\\
  R_0 \\mid \\mathbf{e} &\\sim W^{-1}\\left(\\nu_e + N, \\; V_e + \\sum_k \\mathbf{e}_k
      \\mathbf{e}_k'\\right)
  \\end{aligned}
  ```

# Data Augmentation
For missing phenotypes (e.g., individual 1 missing trait 2):
```math
y_{i, 2} \\mid y_{i, 1}, \\mathbf{b}, \\mathbf{u}, R_0 \\sim N\\left(\\mu_{2\\mid 1}, \\;
    \\sigma_{e_{2\\mid 1}}^2\\right)
```
Missing values are sampled at every Gibbs iteration, allowing straightforward
complete-data matrix updates.

# Tables
The data are one table, one row per animal, with both traits; BLUP and posterior means
are added as columns. For the missing record the conditional distribution used for
data augmentation is computed at the BLUP solutions.

# Returns
A `NamedTuple` with field `ped` (the data with BLUP and posterior means) and `f`: the chain with (co)variances sampled (`GibbsResult`:
posterior means of the location parameters, ``R_0`` and ``G_0``, and the variance samples).
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
