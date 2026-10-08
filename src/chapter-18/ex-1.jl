"""
    FixedNormals(z, i)

A deterministic pseudorandom number generator subtype of `AbstractRNG` that replays
a pre-set vector `z` of standard normal deviates ``N(0, 1)``, used to replicate exact
textbook sample iterations from Mrode & Pocrnic (2023).
"""
mutable struct FixedNormals <: AbstractRNG
    z::Vector{Float64}
    i::Int
end
Random.randn(r::FixedNormals) = r.z[r.i+=1]

"""
    mc_check(label, means, target)

Compare posterior means across independent MCMC chains (columns of `means`)
with an analytical `target` (e.g., BLUP solutions).

Computes the Monte Carlo standard error (MC SE) from between-chain variation:
```math
\\mathrm{SE}_{\\mathrm{MC}} = \\frac{s_{\\text{chains}}}{\\sqrt{C}}
```
and evaluates the standardized discrepancy:
```math
z = \\max_j \\frac{|\\bar{\\mu}_j - \\mathrm{target}_j|}{\\mathrm{SE}_{\\mathrm{MC}, j}}
```
Passes if ``z < 4``.
"""
function mc_check(label, means, target)
    μ = vec(mean(means; dims = 2))
    se = vec(std(means; dims = 2)) ./ sqrt(size(means, 2))
    z = maximum(abs.(μ - target) ./ se)
    ok = z < 4
    @printf(
        "  %-48s %s (max |z| = %.2f, max MC SE = %.3f)\n",
        label,
        ok ? "consistent" : "INCONSISTENT",
        z,
        maximum(se)
    )
    NCHECKS[] += 1
    ok || push!(FAILED, label)
    ok
end

"""
    ex_18_1() -> NamedTuple

Run Example 18.1 from Mrode & Pocrnic (2023): Gibbs sampling for a univariate animal model.

# Sampling Scheme
For the animal model ``\\mathbf{y} = X\\mathbf{b} + Z\\mathbf{a} + \\mathbf{e}``:
1. **Location parameters** ``\\boldsymbol{\\theta} = [\\mathbf{b}' \\; \\mathbf{a}']'``:
   Fully conditional distributions are univariate normal:
   ```math
   \\theta_i \\mid \\boldsymbol{\\theta}_{-i}, \\mathbf{y}, \\sigma_e^2, \\sigma_a^2 \\sim
       N(\\tilde{\\theta}_i, \\, C_{ii}^{-1} \\sigma_e^2)
   ```
   where ``\\tilde{\\theta}_i = C_{ii}^{-1} \\left(r_i - \\sum_{j \\neq i} C_{ij}
   \\theta_j\\right)`` and ``C`` is the MME coefficient matrix.
2. **Variance components**:
   Under scaled inverse-``\\chi^2`` conjugate priors:
   ```math
   \\begin{aligned}
   \\sigma_a^2 \\mid \\mathbf{a} &\\sim \\text{Scale-Inv-}\\chi^2\\left(\\nu_a + q, \\;
       \\frac{\\nu_a S_a^2 + \\mathbf{a}' A^{-1} \\mathbf{a}}{\\nu_a + q}\\right) \\\\
   \\sigma_e^2 \\mid \\mathbf{e} &\\sim \\text{Scale-Inv-}\\chi^2\\left(\\nu_e + N, \\;
       \\frac{\\nu_e S_e^2 + \\mathbf{e}' \\mathbf{e}}{\\nu_e + N}\\right)
   \\end{aligned}
   ```

# Execution Steps
- **Round 1 Replay**: Uses textbook random normal deviates to verify exact arithmetic.
-  **Fixed-Variance Verification**: Confirms that posterior means with variances fixed match
  BLUP solutions.
-  **Full Chain**: Samples location and dispersion parameters with proper
  inverse-``\\chi^2`` priors.

# Returns
A `NamedTuple` with fields:
- `x`: Location parameters after round 1 with the book's random numbers.
- `means`: Posterior means with variances fixed (one column per independent chain).
- `f`: Full Gibbs chain (`GibbsResult`: posterior means and variance samples).
"""
function ex_18_1()
    println("Example 18.1: Gibbs sampling, univariate animal model")

    # Step 1: Pedigree and phenotype data
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    y = [4.5, 2.9, 3.9, 3.5, 5.0]
    X, _ = incidence(sex; levels = ["M", "F"])
    Ai = ainv(ped)
    a = RandomEffect(incidence(calf, 8), Ai, 20.0)

    # Step 2: Round 1 with Mrode's random numbers, starting from b = u = 0
    m = mme(y, X, [a]; σ²e = 40.0)
    rng = FixedNormals(
        [
            0.5855,
            0.7095,
            -0.1093,
            -0.4535,
            0.6059,
            -1.8180,
            0.6301,
            -0.2762,
            -0.2842,
            -0.9193,
        ],
        0,
    )
    x = gibbs_sweep!(zeros(10), m.lhs, m.rhs; scale = 40.0, rng)
    check(
        "round 1: b and u",
        x,
        [
            6.4714,
            6.5728,
            -0.3610,
            -1.3438,
            2.2519,
            -5.8480,
            2.2921,
            -2.1022,
            -2.8204,
            -2.8346,
        ];
        atol = 2e-3,
    )
    e = y - [X a.Z] * x
    check(
        "residuals and ê'ê",
        [e; e'e],
        [3.877, -5.965, -0.571, -0.151, 1.363, 52.816];
        atol = 5e-3,
    )
    check("u'A⁻¹u", x[3:end]' * Ai * x[3:end], 78.819; atol = 0.02)

    # Step 3: With variances fixed, posterior means converge to BLUP solutions
    bl = solve_mme(m)
    means = reduce(
        hcat,
        [
            gibbs(
                y,
                X,
                [a];
                σ²e = 40.0,
                estimate_variances = false,
                niter = 50_000,
                burnin = 1_000,
                rng = Xoshiro(c),
            ).x for c = 1:8
        ],
    )
    mc_check("posterior means (variances fixed) vs BLUP", means, bl)

    # Step 4: Full Gibbs chain
    # With uniform priors (νe = νu = −2, s² = 0) and only five records, the joint
    # posterior is improper near σ²a = 0 and the chain is absorbed there; proper
    # scaled inverse-χ² priors (ν = 4, prior values 40 and 20) are used instead.
    f = gibbs(
        y,
        X,
        [a];
        σ²e = 40.0,
        νe = 4,
        Se = 40.0,
        νu = [4],
        Vu = [fill(4 * 20.0, 1, 1)],
        niter = 100_000,
        burnin = 5_000,
        rng = Xoshiro(2),
    )
    q = mapslices(v -> quantile(v, [0.025, 0.5, 0.975]), f.samples; dims = 2)
    show_table(
        parameter = ["σ²e", "σ²a"],
        median = q[:, 2],
        lower = q[:, 1],
        upper = q[:, 3],
    )
    println(
        "  (5 records: the posteriors are very diffuse; means are not shown in the",
        " book)",
    )
    (; x, means, f)
end
