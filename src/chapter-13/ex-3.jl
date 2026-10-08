"""
    gdmodel(f::Bool) -> NamedTuple

Helper function fitting the genomic additive and dominance model.
If `f == true`, individual genomic inbreeding (proportion of homozygous loci)
is fitted as an extra fixed regression covariate to estimate inbreeding depression.
"""
function gdmodel(f::Bool)
    (; M, pig, pen, ww, σ²a, σ²d, σ²e) = data_13_3()
    n, nsnp = size(M)
    p = vec(mean(M; dims = 1)) ./ 2
    G = grm(Matrix{Int8}(M'), p)
    D = grm(Matrix{Int8}(M'), p; method = :dominance)
    X, _ = incidence(pen)
    if f                     # proportion of homozygous SNPs, f = 1 - h/N
        fi = 1 .- vec(count(==(1), M; dims = 2)) ./ nsnp
        X = [X fi[pig]]
    end
    Z = incidence(pig, n)
    m = mme(
        ww,
        X,
        [RandomEffect(Z, inv(G + 0.01I), σ²a), RandomEffect(Z, inv(D + 0.01I), σ²d)];
        σ²e,
    )
    b, (a, d) = solutions(m, solve_mme(m))
    (; p, G, D, b, a = vec(a), d = vec(d))
end

"""
    ex_13_3() -> NamedTuple

Examples 13.3 & 13.4: Genomic additive and dominance models and inbreeding depression
(Mrode & Pocrnic, 2023, Section 13.4; Vitezica et al., 2013).

# Model Formulation
1. **Genomic Additive and Dominance Model** (Example 13.3):
   ```math
   y = Xb + Z a_g + Z d_g + e
   ```
   where:
   - ``G`` is the genomic additive relationship matrix (VanRaden method 1).
   - ``D`` is the genomic dominance relationship matrix (Vitezica et al., 2013):
     under Hardy–Weinberg equilibrium the dominance coding is orthogonal to the additive
     coding, so the additive and dominance variances are separable and
     ``\\mathrm{cov}(a_g, d_g) = 0``.
2.  **Inbreeding Depression Covariate** (Example 13.4): Adds the individual genomic
   inbreeding coefficient ``f_i = 1 - \\mathrm{het}_i / m`` as a covariate to account for
   directional dominance (inbreeding depression):
   ```math
   y = Xb + f_i \\beta_{\\mathrm{inb}} + Z a_g + Z d_g + e
   ```

# Returns
A `NamedTuple` with fields:
- `r`: Results from model without inbreeding covariate (Example 13.3).
- `s`: Results from model with inbreeding covariate (Example 13.4).
"""
function ex_13_3()
    println("Example 13.3: genomic additive and dominance model")

    # 1. Model without inbreeding covariate (Example 13.3)
    r = gdmodel(false)
    check(
        "allele frequencies",
        r.p,
        [
            0.833,
            0.667,
            0.100,
            0.067,
            0.667,
            0.033,
            0.833,
            0.500,
            0.767,
            0.967,
            0.033,
            0.333,
            0.400,
            0.233,
            0.700,
            0.733,
            0.033,
            0.033,
            0.267,
            0.233,
        ],
    )
    check(
        "G, last row",
        r.G[15, :],
        [
            0.322,
            -0.094,
            -0.568,
            -0.625,
            0.414,
            -0.556,
            0.149,
            0.426,
            -0.186,
            -0.475,
            0.414,
            0.149,
            -0.498,
            0.102,
            1.027,
        ],
    )
    check(
        "D, last row",
        r.D[15, :],
        [
            -0.094,
            0.275,
            -0.092,
            0.270,
            0.136,
            -0.099,
            -0.188,
            -0.170,
            0.016,
            0.281,
            0.077,
            -0.188,
            0.141,
            0.052,
            0.865,
        ],
    )
    show_table(animal = 1:15, additive = r.a, dominance = r.d)
    check("pen", r.b, [17.451, 20.812])
    check(
        "additive",
        r.a,
        [
            -0.095,
            0.572,
            -0.947,
            0.228,
            0.028,
            1.556,
            0.744,
            -0.932,
            -1.315,
            -2.093,
            1.395,
            0.730,
            0.378,
            -0.876,
            0.626,
        ],
    )
    check(
        "dominance",
        r.d,
        [
            0.197,
            0.925,
            -1.007,
            0.504,
            0.087,
            0.369,
            -0.523,
            -0.837,
            -0.744,
            -0.361,
            1.107,
            -0.536,
            1.798,
            0.504,
            1.131,
        ],
    )

    # 2. Model with genomic inbreeding as covariate for inbreeding depression (Example 13.4)
    println("Example 13.4: … with genomic inbreeding as a covariate")
    s = gdmodel(true)
    check("pen and inbreeding depression", s.b, [19.933, 23.518, -3.767])
    check(
        "additive",
        s.a,
        [
            -0.125,
            0.460,
            -0.842,
            0.104,
            -0.167,
            1.442,
            0.577,
            -0.816,
            -1.043,
            -1.830,
            1.385,
            0.562,
            0.407,
            -0.721,
            0.608,
        ],
    )
    check(
        "dominance",
        s.d,
        [
            0.131,
            0.946,
            -0.996,
            0.540,
            -0.020,
            0.381,
            -0.727,
            -0.877,
            -0.669,
            -0.215,
            1.022,
            -0.741,
            1.747,
            0.488,
            1.265,
        ],
    )

    (; r, s)
end
