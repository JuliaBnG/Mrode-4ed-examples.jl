"""
    ex_10_2() -> NamedTuple

Example 10.2: Random regression test-day model for dairy cattle fat yield
(Mrode & Pocrnic, 2023, Section 10.3; Schaeffer & Jamrozik, 1996).

# Model Formulation
Random regression models allow the trajectory of individual genetic merit and permanent
environment to vary dynamically across the lactation:
```math
y_{tijk} = \\mathrm{HTD}_i + \\sum_{r=0}^4 \\beta_r \\phi_r(d) +
\\sum_{r=0}^2 a_{jr} \\phi_r(d) + \\sum_{r=0}^2 pe_{jr} \\phi_r(d) + e_{tijk}
```
where:
- Fixed lactation curve has order 4 (Legendre polynomials ``\\phi_0`` to ``\\phi_4``).
- Random animal additive genetic regressions ``a_j`` and permanent environment regressions
  ``pe_j`` have order 2 (Legendre polynomials ``\\phi_0`` to ``\\phi_2``).
- Genetic covariance matrix for regression coefficients: ``G`` (3 × 3).
- Permanent environmental covariance matrix: ``P`` (3 × 3).
- Residual variance: ``\\sigma_e^2 = 3.710``.

# Key Concepts Illustrated
1. **Tables**: the test-day records of Example 10.1, one row per test day;
   regression coefficients, 305-day values and reliabilities are added as
   columns of the pedigree.
2. **Random Regression Design Matrices**: Constructed with diagonal covariable weights
   `Diagonal(ϕ[:, c]) * Z`. The MME are those of Example 10.1, ``T'T + \\Sigma``,
   with ``Z_{RR} = [\\mathrm{Diag}(\\phi_0)Z\\ \\mathrm{Diag}(\\phi_1)Z\\
   \\mathrm{Diag}(\\phi_2)Z]`` in place of ``Z`` and ``\\sigma_e^2 G^{-1} \\otimes A^{-1}``
   in place of ``\\lambda_a A^{-1}`` (likewise for pe); compared with `mme`.
3. **305-Day Breeding Values and PE**: Summed across lactation days:
   ``t = \\sum_{d=6}^{310} \\phi(d)``, with ``\\mathrm{EBV}_{305} = u t``.
4. **Reliability of Linear Functions of Random Regressions** (Section 10.3.4):
   For evaluation ``k' u`` with ``k = t``:
   ``\\mathrm{rel} = 1 - \\frac{t' C^{ii} t}{t' G t}``.
5. **Decomposition of Random Regressions** (Section 10.3.2):
   Multi-trait partition into PA, YD, and PC across Legendre coefficients.

# Notes on Text Discrepancies
1.  Vector ``t``: The book sums Legendre polynomials built from 4-decimal rounded
   coefficients, causing slight differences in ``t_3`` (-1.5561 vs -1.5451).
2.  The quadratic form ``k' P k`` is printed as 323462.969; the book's own ``t`` and ``P``
   give 320126.

# Returns
A `NamedTuple` with fields:
- `ped`: The pedigree with columns `u0`, `u1`, `u2`, `bv305`, `pe305` and `rel` added.
- `rec`: The test-day records with the corrected yields `yc` added.
- `b`: Fixed effect solutions (HTD and curve).
- `u`: Animal random regression coefficients (8 × 3).
- `pe`: Permanent environmental random regression coefficients (5 × 3).
- `rel`: Reliabilities of 305-day breeding values for all 8 animals.
"""
function ex_10_2()
    println("Example 10.2: random regression model")

    # 1. Test-day records (one row per test day), pedigree, Legendre polynomials,
    #    and the covariance matrices of the random regression coefficients
    (; ped, rec, Φ) = data_10()
    G = [3.297 0.594 -1.381; 0.594 0.921 -0.289; -1.381 -0.289 1.005]
    P = [6.872 -0.254 -1.101; -0.254 3.171 0.167; -1.101 0.167 2.457]
    σ²e = 3.710
    N = nrow(ped)
    Ai = ainv(ped)

    # 2. Design matrices. Fixed effects as in Example 10.1; for the random
    #    regressions, record i gets the covariables ϕ(dᵢ) of its DIM in the
    #    columns of its cow: Z_RR = [Diag(ϕ₀)Z Diag(ϕ₁)Z Diag(ϕ₂)Z]
    y = rec.tdy
    X1, _ = incidence(rec.htd)
    X2 = Φ[rec.dim, :]
    Z = incidence(rec.cow, N)
    Q, cows = incidence(rec.cow)
    ϕ = Φ[rec.dim, 1:3]                   # order-2 Legendre covariables for random effects
    ZR = reduce(hcat, [Diagonal(ϕ[:, c]) * Z for c = 1:3])
    QR = reduce(hcat, [Diagonal(ϕ[:, c]) * Q for c = 1:3])

    # 3. MME (R = Iσ²e) as in Example 10.1, T'T + Σ, with T = [X₁ X₂ Z_RR Q_RR]
    #    and the scalar ratios replaced by σ²e G⁻¹ ⊗ A⁻¹ and σ²e P⁻¹ ⊗ I
    T = [X1 X2 ZR QR]
    nb = size(X1, 2) + size(X2, 2)
    Σ = blockdiag(
        spzeros(nb, nb),
        sparse(kron(σ²e * inv(G), Ai)),
        sparse(kron(σ²e * inv(P), I(length(cows)))),
    )
    lhs = T'T + Σ
    rhs = T'y
    m = mme(
        y,
        [X1 X2],
        [
            RandomEffect([Diagonal(ϕ[:, c]) * Z for c = 1:3], Ai, G),
            RandomEffect([Diagonal(ϕ[:, c]) * Q for c = 1:3], I, P),
        ];
        σ²e,
    )
    check("mme() LHS = T'T + Σ", m.lhs, lhs; atol = 1e-12)

    # 4. Solve MME (HTD 10 set to zero)
    x = solve_mme(lhs, rhs; zero = [10])
    b = x[m.blocks[1]]
    u = reshape(x[m.blocks[2]], :, 3)     # animals × (ϕ₀, ϕ₁, ϕ₂)
    pe = reshape(x[m.blocks[3]], :, 3)    # cows × (ϕ₀, ϕ₁, ϕ₂)

    t106, t140 = legendre([106, 140], 3; tmin = 4, tmax = 310) |> eachrow
    check(
        "genetic variance at DIM 106, covariance 106–140",
        [t106' * G * t106, t106' * G * t140],
        [2.6433, 3.0219];
        atol = 1e-3,
    )
    check(
        "HTD",
        b[1:10],
        [10.0862, 7.5908, 8.5601, 8.2430, 6.3161, 3.0101, 3.1085, 3.1718, 0.5044, 0];
        atol = 2e-4,
    )
    check(
        "fixed regressions",
        b[11:15],
        [16.6384, -0.6253, -0.1346, 0.3479, -0.4218];
        atol = 1e-4,
    )
    check(
        "animal regressions",
        u,
        [
            -0.0583 0.0552 -0.0442;
            -0.0728 -0.0305 -0.0244;
            0.1311 -0.0247 0.0686;
            0.3445 0.0063 -0.3164;
            -0.4537 -0.0520 0.2798;
            -0.5485 0.0730 0.1946;
            0.8518 -0.0095 -0.3131;
            0.2209 0.0127 -0.0174
        ];
        atol = 1e-4,
    )
    check(
        "pe regressions",
        pe,
        [
            -0.6487 -0.3601 -1.4718;
            -0.7761 0.1370 0.9688;
            -1.9927 0.9851 -0.0693;
            3.5188 -1.0510 -0.4048;
            -0.1013 0.2889 0.9771
        ];
        atol = 1e-4,
    )

    # 5. 305-day cumulative breeding values: t = Σ_{d=6}^{310} φ(d)
    t = vec(sum(legendre(6:310, 3; tmin = 4, tmax = 310); dims = 1))
    check("t (days 6–310)", t, [215.6655, 2.4414, -1.5561]; atol = 0.012)
    ped.u0, ped.u1, ped.u2 = eachcol(u)
    ped.bv305 = u * t
    ped.pe305 = Vector{Union{Missing,Float64}}(missing, N)
    ped.pe305[cows] = pe * t
    check(
        "305-day breeding values",
        u * t,
        [-12.3731, -15.7347, 28.1078, 74.8132, -98.4153, -118.4265, 184.1701, 47.6907];
        atol = 0.03,
    )
    check(
        "305-day pe",
        pe * t,
        [-138.4887, -168.5531, -427.2378, 756.9415, -22.6619];
        atol = 0.03,
    )

    # 6. Reliability of 305-day breeding values k'u (k = t): PEV = t' Cⁱⁱ t (Section 10.3.4)
    C = lhs_inverse(m; zero = [10]) * m.scale
    r = m.blocks[2]
    g = t' * G * t
    rel = [1 - (idx = r[(0:2) .* N .+ i]; t' * C[idx, idx] * t) / g for i = 1:N]
    ped.rel = rel
    check("k'Gk", g, 154896.766; atol = 25)
    @printf("  k'Pk = %.1f\n", t' * P * t)
    check(
        "reliabilities",
        rel,
        [0.09, 0.04, 0.07, 0.12, 0.15, 0.06, 0.10, 0.05];
        atol = 0.005,
    )

    # 7. Partition of random regression coefficients (Section 10.3.2): records
    #    corrected for HTD, curve and pe, and per cow the least-squares
    #    regression of the corrected records on ϕ (the YD of the coefficients)
    rec.yc = y - [X1 X2] * b - QR * vec(pe)
    ZRZ = [zeros(3, 3) for _ = 1:N]
    yd = zeros(N, 3)
    for g in groupby(rec, :cow)
        i, ϕg = g.cow[1], Φ[g.dim, 1:3]
        ZRZ[i] = ϕg'ϕg / σ²e
        yd[i, :] = (ϕg'ϕg) \ (ϕg' * g.yc)
    end
    p = ebv_partition_mt(ped.sire, ped.dam, G, ZRZ, yd, u)

    check("YD of cow 6", yd[6, :], [-5.0004, -4.6419, -1.9931]; atol = 5e-3)
    check("YD of cow 6, 305-day value", t' * yd[6, :], -1086.6450; atol = 0.8)
    check("cow 6: W₁ (row 2)", p.W1[6][2, :], [0.0935, 0.8107, 0.2402]; atol = 1e-4)
    check("cow 6: W₂ (row 2)", p.W2[6][2, :], [-0.0935, 0.1893, -0.2402]; atol = 1e-4)
    PAs = sum(Diagonal(t) * p.W1[6] * p.PA[6])
    YDs = sum(Diagonal(t) * p.W2[6] * yd[6, :])
    check(
        "cow 6: 305-day contributions of PA and YD",
        [PAs, YDs],
        [-26.4049, -91.9862];
        atol = 0.05,
    )
    check(
        "cow 4: W₁PA + W₂YD + W₃PC = û₄",
        p.W1[4] * p.PA[4] + p.W2[4] * yd[4, :] + p.W3[4] * p.PC[4],
        u[4, :];
        atol = 1e-10,
    )

    println("\nPedigree with random regression coefficients, 305-day values and reliabilities:")
    show_table(ped; digits = 4)

    (; ped, rec, b, u, pe, rel)
end
