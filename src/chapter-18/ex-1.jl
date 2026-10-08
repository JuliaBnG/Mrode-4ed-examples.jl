# Example 18.1 – Gibbs sampling for the univariate animal model: the
# first round replayed with the book's normal deviates, then full chains.

"""An RNG that returns pre-set N(0, 1) deviates (to replay the book)."""
mutable struct FixedNormals <: AbstractRNG
    z::Vector{Float64}
    i::Int
end
Random.randn(r::FixedNormals) = r.z[r.i += 1]

"""
    mc_check(label, means, target)

Compare posterior means from independent chains (columns of `means`)
with `target`: report and record (as [`check`](@ref)) the largest |z| = |mean − target| / MC SE, with the
Monte Carlo SE from the between-chain spread. Passes if |z| < 4.
"""
function mc_check(label, means, target)
    μ = vec(mean(means; dims = 2))
    se = vec(std(means; dims = 2)) ./ sqrt(size(means, 2))
    z = maximum(abs.(μ - target) ./ se)
    ok = z < 4
    @printf("  %-48s %s (max |z| = %.2f, max MC SE = %.3f)\n", label,
            ok ? "consistent" : "INCONSISTENT", z, maximum(se))
    NCHECKS[] += 1
    ok || push!(FAILED, label)
    ok
end

function ex_18_1()
    println("Example 18.1: Gibbs sampling, univariate animal model")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    y = [4.5, 2.9, 3.9, 3.5, 5.0]
    X, _ = incidence(sex; levels = ["M", "F"])
    Ai = ainv(ped)
    a = RandomEffect(incidence(calf, 8), Ai, 20.0)

    # round 1 with Mrode's random numbers, starting from b = u = 0
    m = mme(y, X, [a]; σ²e = 40.0)
    rng = FixedNormals([0.5855, 0.7095, -0.1093, -0.4535, 0.6059, -1.8180, 0.6301, -0.2762,
                        -0.2842, -0.9193], 0)
    x = gibbs_sweep!(zeros(10), m.lhs, m.rhs; scale = 40.0, rng)
    check("round 1: b and u", x, [6.4714, 6.5728, -0.3610, -1.3438, 2.2519, -5.8480, 2.2921,
                                  -2.1022, -2.8204, -2.8346]; atol = 2e-3)
    e = y - [X a.Z] * x
    check("residuals and ê'ê", [e; e'e], [3.877, -5.965, -0.571, -0.151, 1.363, 52.816];
          atol = 5e-3)
    check("u'A⁻¹u", x[3:end]' * Ai * x[3:end], 78.819; atol = 0.02)

    # with variances fixed, posterior means converge to the BLUP solutions
    bl = solve_mme(m)
    means = reduce(hcat, [gibbs(y, X, [a]; σ²e = 40.0, estimate_variances = false,
                                niter = 50_000, burnin = 1_000, rng = Xoshiro(c)).x
                          for c in 1:8])
    mc_check("posterior means (variances fixed) vs BLUP", means, bl)

    # Full chain. With the book's uniform priors (νe = νu = −2, s² = 0) and only
    # five records the joint posterior is improper near σ²a = 0 and the chain
    # is absorbed there; proper scaled inverse-χ² priors (ν = 4, prior
    # values 40 and 20) are used instead.
    f = gibbs(y, X, [a]; σ²e = 40.0, νe = 4, Se = 40.0, νu = [4], Vu = [fill(4 * 20.0, 1, 1)],
              niter = 100_000, burnin = 5_000, rng = Xoshiro(2))
    q = mapslices(v -> quantile(v, [0.025, 0.5, 0.975]), f.samples; dims = 2)
    show_table(parameter = ["σ²e", "σ²a"], median = q[:, 2], lower = q[:, 1],
               upper = q[:, 3])
    println("  (5 records: the posteriors are very diffuse; means are not shown in the",
            " book)")
    (; x, means, f)
end
