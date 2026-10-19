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

Example 18.1: Gibbs sampling for a univariate animal model (Mrode & Pocrnic, 2023,
Section 18.2; Wang et al., 1994).

# Model Formulation
For the univariate animal model ``y = Xb + Za + e`` with MME coefficient matrix ``C`` and
right-hand side ``r`` (in the ``\\lambda = \\sigma_e^2/\\sigma_a^2`` form):
1. **Location parameters** ``\\theta = [b;\\ a]``: Each unknown ``\\theta_i`` has a
   univariate normal full conditional distribution given all other parameters
   ``\\theta_{-i}``:
   ```math
   \\theta_i \\mid \\theta_{-i}, y, \\sigma_e^2, \\sigma_a^2 \\sim
       N\\!\\left(\\tilde{\\theta}_i,\\; \\frac{\\sigma_e^2}{C_{ii}}\\right), \\quad
   \\tilde{\\theta}_i = \\frac{1}{C_{ii}}\\left(r_i - \\sum_{j \\neq i} C_{ij} \\theta_j\\right).
   ```
2. **Variance components**: Under scaled inverse-``\\chi^2`` conjugate priors
   ``(\\nu_a, S_a^2)`` and ``(\\nu_e, S_e^2)``, the full conditionals are:
   ```math
   \\begin{aligned}
   \\sigma_a^2 \\mid a &\\sim \\chi^{-2}\\!\\left(\\nu_a + q, \\;
       \\frac{\\nu_a S_a^2 + a' A^{-1} a}{\\nu_a + q}\\right), \\\\
   \\sigma_e^2 \\mid e &\\sim \\chi^{-2}\\!\\left(\\nu_e + N, \\;
       \\frac{\\nu_e S_e^2 + e' e}{\\nu_e + N}\\right).
   \\end{aligned}
   ```

# Key Concepts Illustrated
1. **Round 1 written out as a single-site loop**: Using the 8-animal data table of
   [`ex_04_1`](@ref), we write out one full sweep of single-site updates from ``\\theta = 0``
   using the textbook's 10 standard normal deviates (`FixedNormals`) and verify that it
   matches `gibbs_sweep!` and the book's Round 1 numbers.
2. **Convergence of posterior means to BLUP when variances are fixed**: Running 8 independent
   MCMC chains with ``\\sigma_a^2 = 20`` and ``\\sigma_e^2 = 40`` fixed confirms that the
   posterior mean of ``\\theta`` equals the analytical MME solution ``C^{-1}r`` within Monte
   Carlo standard error.
3. **Full Gibbs sampling and proper priors**: With flat priors (``\\nu_e = \\nu_a = -2,
   S^2 = 0``) and only 5 records, the joint posterior is improper near ``\\sigma_a^2 = 0``
   and the chain is absorbed there; proper scaled inverse-``\\chi^2`` priors (``\\nu = 4``)
   are used for the full chain, and all three sets of animal estimates (`blup`,
   `gibbs_fixed`, `gibbs_full`) are compared side by side in `ped`.

# Returns
A `NamedTuple` with fields:
- `x`: Location parameters after Round 1 using the book's random normal deviates.
- `means`: Posterior means with variances fixed (one column per independent chain).
- `f`: Full Gibbs chain with variances sampled (`GibbsResult`).
- `ped`: The data table with `blup`, `gibbs_fixed`, and `gibbs_full` added.
"""
function ex_18_1()
    println("Example 18.1: Gibbs sampling, univariate animal model")

    # Step 1: Pedigree and records of Example 4.1, one row per animal
    no3 = fill(missing, 3)
    ped = DataFrame(
        calf = 1:8,
        sex = [no3; ["M", "F", "F", "M", "M"]],
        sire = [0, 0, 0, 1, 3, 1, 4, 3],
        dam = [0, 0, 0, 0, 2, 2, 5, 6],
        wwg = [no3; [4.5, 2.9, 3.9, 3.5, 5.0]],
    )
    rec = dropmissing(ped, :wwg)
    y = rec.wwg
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Z = incidence(rec.calf, nrow(ped))
    Ai = ainv(ped)
    a = RandomEffect(Z, Ai, 20.0)

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
    #    written out: one sweep of single-site updates, each θᵢ drawn from
    #    N((rᵢ - Σⱼ≠ᵢ Cᵢⱼθⱼ)/Cᵢᵢ, σ²e/Cᵢᵢ) given the current values of the others
    C, r = Matrix(m.lhs), m.rhs
    θ = zeros(10)
    for i = 1:10
        μi = (r[i] - C[i, :]' * θ + C[i, i] * θ[i]) / C[i, i]
        θ[i] = μi + rng.z[i] * sqrt(40.0 / C[i, i])
    end
    check("round 1 written out = gibbs_sweep!()", θ, x; atol = 1e-12)
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
    e = y - [X Z] * x
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

    ped.blup = bl[3:end]
    ped.gibbs_fixed = vec(mean(means; dims = 2))[3:end]
    ped.gibbs_full = f.x[3:end]
    println("\nBLUP, posterior means with variances fixed, and with variances sampled:")
    show_table(ped)

    (; x, means, f, ped)
end
