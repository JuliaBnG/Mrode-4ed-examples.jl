"""
    ex_09_1() -> NamedTuple

Example 9.1: Animal model with social interaction (indirect genetic / associative) effects
(Mrode & Pocrnic, 2023, Chapter 9; Bijma et al., 2007; Muir, 2005).

# Model Formulation
When animals are housed in groups (e.g. pigs in pens of three), an individual's phenotype
is affected by its own genes (direct breeding value, DBV) and the genes of its pen-mates
(social/associative breeding value, SBV):
```math
y_i = x_i' b + u_{Di} + \\sum_{j \\neq i} u_{Sj} + g_k + c_l + e_i^*
```
where:
- ``u_{Di}`` is the direct genetic effect of animal ``i``, with variance ``\\sigma_{AD}^2``.
-  ``u_{Sj}`` is the associative genetic effect of pen-mate ``j``, with variance
  ``\\sigma_{AS}^2``.
-  Direct and associative genetic effects are correlated:
  ``\\mathrm{cov}(u_D, u_S) = \\sigma_{ADS}``, so ``G_0 = \\begin{bmatrix} \\sigma_{AD}^2 &
  \\sigma_{ADS} \\\\ \\sigma_{ADS} & \\sigma_{AS}^2 \\end{bmatrix}``.
- Total Breeding Value (TBV) is the overall effect on the group mean:
  ``\\mathrm{TBV}_i = u_{Di} + (n - 1) u_{Si}`` (here ``n = 3``).
-  ``g_k`` is the random group (pen) environmental effect, with variance
  ``\\sigma_g^2 = \\rho \\sigma_e^2``.
- ``c_l`` is the common litter environmental effect, with variance ``\\sigma_c^2``.

# Key Concepts Illustrated
1. **One table for the data**: pedigree, sex, pen, litter and growth rate of
   all 15 pigs in one `DataFrame`; DBV, SBV, TBV and the EBV of the model
   without associative effects are added as columns, and the pen and litter
   solutions kept in tables of their own.
2. **Associative Incidence Matrix**: ``Z_S = (VV' - I) Z_D``, where ``VV'``
   marks records in the same pen; compared with
   `associate_incidence(pen, pig, N)`.
3. **MME** (Eqn 9.6) as ``T'T + \\Sigma`` with ``T = [X\\ Z_D\\ Z_S\\ V\\ W]`` and
   ``\\Sigma`` holding ``\\alpha \\otimes A^{-1}`` (``\\alpha = \\sigma_{e^*}^2 G_0^{-1}``) and the
   variance ratios of the group and litter effects; compared with `mme`.
4. **Comparison with Standard Animal Model**: Animal model ignoring associative effects.
5. **Equivalence with Correlated Residuals Model** (Section 9.4): Instead of fitting an
   explicit random group effect ``g``, pen-mates have correlated residuals:
   ``R = \\sigma_e^2 [(1 - \\rho) I + \\rho VV']``, block diagonal
   ``R_i = \\sigma_e^2 [(1 - \\rho) I_n + \\rho J_n]`` per pen, yielding identical
   fixed and genetic solutions.
6. **Decomposition of DBV and SBV** (Section 9.3):
   Partition into Parent Average (PA) and direct/associative yield deviations (YD).

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `DBV`, `SBV`, `TBV` and `ebv_noassoc` added.
- `pens`, `litters`: The group and litter solutions.
- `b`: Fixed sex effects.
- `u`: Direct (col 1) and social (col 2) genetic solutions for all 15 animals.
- `g`: Random pen group effects.
- `c`: Random common litter effects.
"""
function ex_09_1()
    println("Example 9.1: growth rate of pigs in pens of three")

    # 1. Pedigree, pen allocation and records, one row per pig
    #    (IDs are row numbers; parents 1–6 have no records)
    no6 = fill(missing, 6)
    ped = DataFrame(
        pig = 1:15,
        sire = [0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 3, 2, 3, 3],
        dam = [0, 0, 0, 0, 0, 0, 4, 4, 5, 4, 5, 6, 5, 6, 6],
        sex = [no6; ["M", "F", "F", "M", "F", "F", "M", "F", "M"]],
        pen = [no6; [1, 1, 1, 2, 2, 2, 3, 3, 3]],
        litter = [no6; [1, 1, 2, 1, 2, 3, 2, 3, 3]],
        gr = [no6; [5.50, 9.80, 4.90, 8.23, 7.50, 10.0, 4.50, 8.40, 6.40]],
    )
    @assert ped.pig == 1:nrow(ped)
    rec = dropmissing(ped, :gr)                 # the pigs with records
    println(ped)

    # 2. Variance components
    G0 = [25.70 2.25; 2.25 3.60]               # direct, associative genetic covariance
    σ²c, σ²ED, σ²ES, ρ, n = 12.5, 40.6, 10.0, 0.2, 3
    σ²e = σ²ED + (n - 1) * σ²ES                # total residual variance: 60.6
    σ²g = ρ * σ²e                              # group (pen) variance: 12.12
    σ²e_star = σ²e - σ²g                       # residual variance after pen effect: 48.48
    α = σ²e_star * inv(G0)
    check("α = σ²e* G₀⁻¹", α, [1.9956 -1.2472; -1.2472 14.2462])

    # 3. Incidence matrices. A record is linked to the pig's own direct effect
    #    (Z_D) and to the associative effects of its pen-mates (Z_S): V V' marks
    #    records in the same pen, minus I removes the pig itself.
    N = nrow(ped)
    Ai = ainv(ped)
    y = rec.gr
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    ZD = incidence(rec.pig, N)
    V, _ = incidence(rec.pen)
    W, _ = incidence(rec.litter)
    ZS = (V * V' - I) * ZD
    check("Z_S = associate_incidence()", ZS, associate_incidence(rec.pen, rec.pig, N); atol = 0)
    check("Z_S (pen-mates)", Matrix(ZS[:, rec.pig]), kron(I(3), ones(3, 3) - I(3)); atol = 0)

    # 4. MME (Eqn 9.6), R = Iσ²e*: with T = [X Z_D Z_S V W], the coefficient
    #    matrix is T'T plus the variance ratios of the random effects, i.e. the
    #    block form [X'X X'Z_D …; Z_D'X Z_D'Z_D + α₁₁A⁻¹ Z_D'Z_S + α₁₂A⁻¹ …; …]
    T = [X ZD ZS V W]
    Σ = blockdiag(
        spzeros(size(X, 2), size(X, 2)),
        sparse(kron(α, Ai)),                    # direct and associative, correlated
        sparse(σ²e_star / σ²g * I, size(V, 2), size(V, 2)),
        sparse(σ²e_star / σ²c * I, size(W, 2), size(W, 2)),
    )
    lhs = T'T + Σ
    rhs = T'y

    m = mme(
        y,
        X,
        [
            RandomEffect([ZD, ZS], Ai, G0; name = "direct+associative"),
            RandomEffect(V, I, σ²g; name = "group"),
            RandomEffect(W, I, σ²c; name = "litter"),
        ];
        σ²e = σ²e_star,
    )
    check("mme() LHS = T'T + Σ", m.lhs, lhs; atol = 1e-12)

    # 5. Solve MME and evaluate TBV
    x = lhs \ rhs
    b = x[m.blocks[1]]
    u = reshape(x[m.blocks[2]], :, 2)           # rows: pigs, cols: direct and associative
    g = x[m.blocks[3]]
    c = x[m.blocks[4]]
    tbv = u[:, 1] + (n - 1) * u[:, 2]
    ped.DBV, ped.SBV, ped.TBV = u[:, 1], u[:, 2], tbv
    pens = DataFrame(pen = 1:3, g = g)
    litters = DataFrame(litter = 1:3, c = c)

    check("sex", b, [6.004, 8.243])
    check(
        "DBV",
        u[:, 1],
        [
            0.296,
            -0.483,
            0.188,
            0.296,
            -0.483,
            0.188,
            0.125,
            0.522,
            -0.874,
            0.536,
            -0.488,
            0.399,
            -0.572,
            0.153,
            0.199,
        ],
    )
    check(
        "SBV",
        u[:, 2],
        [
            -0.044,
            0.028,
            0.017,
            -0.044,
            0.028,
            0.017,
            -0.076,
            -0.099,
            0.009,
            -0.003,
            0.083,
            0.060,
            0.019,
            0.005,
            0.002,
        ],
    )
    check(
        "TBV",
        tbv,
        [
            0.207,
            -0.428,
            0.221,
            0.207,
            -0.428,
            0.221,
            -0.027,
            0.324,
            -0.856,
            0.530,
            -0.321,
            0.519,
            -0.534,
            0.163,
            0.203,
        ];
        atol = 2e-3,
    )
    check("common environment", c, [0.333, -0.515, 0.183])
    check("group", g, [-0.269, 0.359, -0.090])
    check("non-zeros of the MME (Eqn 9.6)", nnz(m.lhs), 462)

    # 5. Model without associative effects: fixed pen and sex (male = 0)
    m0 = mme(y, [V X], [RandomEffect(ZD, Ai, G0[1, 1]), RandomEffect(W, I, σ²c)]; σ²e)
    x0 = solve_mme(m0; zero = [4])
    b0, (a0, c0) = solutions(m0, x0)
    check("no associative effects: pen and sex", b0, [5.160, 7.131, 5.838, 0, 2.169])
    check(
        "no associative effects: animals",
        a0,
        [
            0.336,
            -0.478,
            0.142,
            0.336,
            -0.478,
            0.142,
            0.279,
            0.652,
            -0.738,
            0.412,
            -0.628,
            0.216,
            -0.547,
            0.162,
            0.192,
        ],
    )
    check("no associative effects: litters", c0, [0.327, -0.465, 0.139])
    ped.ebv_noassoc = vec(a0)

    # 6. Equivalent correlated residuals model (Section 9.4): instead of a group
    #    effect, pen-mates have correlated residuals, R = σ²e [(1 - ρ) I + ρ V V'].
    #    With records sorted by pen this is block diagonal, I ⊗ σ²e [(1 - ρ) I + ρ J].
    R = σ²e * ((1 - ρ) * I + ρ * V * V')
    Rinv = sparse(inv(Matrix(R)))
    Rb = σ²e * ((1 - ρ) * I(3) + ρ * ones(3, 3))
    check("R⁻¹ = I ⊗ Rb⁻¹ (records sorted by pen)", Rinv, kron(I(3), inv(Rb)); atol = 1e-12)
    check("R⁻¹ block", inv(Rb)[1, 1:2], [0.01768, -0.00295]; atol = 1e-5)
    mr = mme(y, X, [RandomEffect([ZD, ZS], Ai, G0), RandomEffect(W, I, σ²c)]; Rinv)
    xr = solve_mme(mr)
    check(
        "Eqn 9.7 = Eqn 9.6 (all solutions but groups)",
        xr,
        [x[1:32]; x[36:38]];
        atol = 1e-8,
    )
    check("non-zeros of the MME (Eqn 9.7)", nnz(mr.lhs), 481)

    # 7. Partition of breeding values for pig 7 (Section 9.3): (ûD, ûS) = WT₁ PA + WT₂ yd
    α = σ²e_star * inv(G0)
    e = y - T * x                               # residuals of the full model
    yd1 = e[1] + u[7, 1]
    yd2 = sum(e[k] + u[7, 2] for k in (2, 3)) / (n - 1)
    D = 2α + Diagonal([1, 2])
    WT1, WT2 = D \ 2α, D \ Diagonal([1, 2])
    PA = (u[1, :] + u[4, :]) / 2
    check("pig 7: yd₁, yd₂", [yd1, yd2], [-0.478, -0.312])
    check("pig 7: DIAG", D, [4.991 -2.494; -2.494 30.492])
    check(
        "pig 7: WT₁, WT₂",
        [WT1 WT2],
        [0.791 -0.034 0.209 0.034; -0.017 0.932 0.017 0.068],
    )
    check(
        "pig 7: WT₁PA + WT₂yd = (ûD, ûS)",
        WT1 * PA + WT2 * [yd1, yd2],
        u[7, :];
        atol = 1e-10,
    )

    println("\nPens and litters:")
    show_table(pens)
    show_table(litters)
    println("\nData with DBV, SBV, TBV, and EBV without associative effects:")
    show_table(ped)

    (; ped, pens, litters, b, u, g, c)
end
