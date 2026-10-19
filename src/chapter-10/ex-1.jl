"""
    ex_10_1() -> NamedTuple

Example 10.1: Fixed regression test-day model for dairy cattle fat yield
(Mrode & Pocrnic, 2023, Sections 10.1–10.2).

# Model Formulation
Test-day records within lactation are modeled with a fixed population lactation curve:
```math
y_{tijk} = \\mathrm{HTD}_i + \\sum_{r=0}^4 \\beta_r \\phi_r(d) + a_j + pe_j + e_{tijk}
```
where:
- ``\\mathrm{HTD}_i`` is the fixed herd-test-date effect (10 levels).
-  ``\\phi_r(d)`` are normalized Legendre polynomials of order 4 (covariables 0 to 4)
  evaluated at days in milk ``d \\in [4, 310]``, with fixed regression coefficients
  ``\\beta``.
- ``a_j`` is the random animal additive genetic effect (constant daily EBV).
- ``pe_j`` is the random permanent environmental effect across test-days for cow ``j``.
-  Variance components: ``\\sigma_a^2 = 5.521``, ``\\sigma_{pe}^2 = 8.470``,
  ``\\sigma_e^2 = 3.710``.

# Key Concepts Illustrated
1. **Data and Result Tables**: Table 10.1 is entered as in the book (one column per cow)
   and stacked with `stack` to one row per test day. Daily and 305-day EBVs, permanent
   environmental effects, yield deviations, and partition weights are added as columns of
   the pedigree `DataFrame`.
2. **MME Formulation**: Assembled as ``T'T + \\Sigma`` with ``T = [X_1\\ X_2\\ Z\\ Q]``, where
   row ``i`` of ``X_2`` is the row of ``\\Phi`` for the DIM of record ``i``, and verified
   against `mme`.
3. **Legendre Polynomials for Lactation Curves**: Orthogonal polynomials evaluated over
   standardized days in milk.
4. **Fixed Regression Test-Day Model**: Fits a single average population curve to all cows,
   with constant daily breeding values and permanent environmental effects per cow.
5. **305-Day Genetic Evaluations**: Daily EBV scaled by 305 days.
6. **Decomposition of EBVs**: Partition into Parent Average (PA) and Yield Deviation (YD)
   adjusted for HTD, lactation curve, and permanent environment.

# Note on Text Discrepancy
The book's printed values for the fixed lactation curve from DIM 174 on (11.0407, 10.9156,
etc.) do not match ``\\Phi \\hat{\\beta}_2`` (e.g. at DIM 174,
``\\Phi_{174} \\hat{\\beta}_2 = 11.099``). The exact curve values are checked and verified
here.

# Returns
A `NamedTuple` with fields:
- `ped`: The pedigree with columns `ebv`, `ebv305`, `pe`, `n`, `YD`, `w_PA` and `w_YD` added.
- `rec`: The test-day records with the corrected yields `yc` added.
- `curve`: The fixed lactation curve at the ten DIM.
- `b`: Fixed effect solutions (HTD and Legendre curve coefficients).
- `a`: Daily estimated breeding values for animals 1 to 8.
- `pe`: Daily permanent environmental effects for cows 4 to 8.
"""
function ex_10_1()
    println("Example 10.1: fixed regression model for test-day fat yield")

    # 1. Test-day records (one row per test day), pedigree and Legendre polynomials
    (; ped, rec, dims, Φ) = data_10()
    σ²a = 5.521
    σ²pe = 8.470
    σ²e = 3.710
    println(rec)

    check(
        "Φ (Appendix G), first and last rows",
        Φ[[1, 10], :],
        [0.7071 -1.2247 1.5811 -1.8704 2.1213; 0.7071 1.2247 1.5811 1.8704 2.1213];
        atol = 5e-4,
    )  # the book rounds the Legendre coefficients to 4 decimals

    # 2. Design matrices: row i of X₂ is the row of Φ for the DIM of record i
    y = rec.tdy
    X1, _ = incidence(rec.htd)                      # fixed HTD effects (10 levels)
    X2 = Φ[rec.dim, :]                              # fixed lactation curve (order 4)
    Z = incidence(rec.cow, nrow(ped))               # random animal effect
    Q, cows = incidence(rec.cow)                    # random pe effect, cows 4–8
    Ai = ainv(ped)
    check(
        "X₁'X₁ = diag(records per HTD)",
        diag(X1'X1),
        [3, 3, 3, 4, 4, 5, 5, 5, 5, 5];
        atol = 0,
    )

    # 3. MME (R = Iσ²e): with T = [X₁ X₂ Z Q], the coefficient matrix is T'T plus
    #    λa A⁻¹ in the animal block and λpe I in the pe block
    T = [X1 X2 Z Q]
    Σ = blockdiag(
        spzeros(size(X1, 2) + size(X2, 2), size(X1, 2) + size(X2, 2)),
        sparse(σ²e / σ²a * Ai),
        sparse(σ²e / σ²pe * I, length(cows), length(cows)),
    )
    lhs = T'T + Σ
    rhs = T'y
    m = mme(y, [X1 X2], [RandomEffect(Z, Ai, σ²a), RandomEffect(Q, I, σ²pe)]; σ²e)
    check("mme() LHS = T'T + Σ", m.lhs, lhs; atol = 1e-12)

    check(
        "X₂'X₂",
        Matrix(X2'X2),
        [
            20.9996 -4.4261 4.0568 -0.8441 8.7149;
            -4.4261 24.6271 -4.7012 11.1628 -3.0641;
            4.0568 -4.7012 31.0621 -6.6603 19.0867;
            -0.8441 11.1628 -6.6603 38.6470 -8.8550;
            8.7149 -3.0641 19.0867 -8.8550 48.2930
        ];
        atol = 1e-2,
    )

    # 4. Solve MME (HTD 10 set to zero)
    x = solve_mme(lhs, rhs; zero = [10])            # HTD 10 set to zero
    b = x[m.blocks[1]]
    a = x[m.blocks[2]]
    pe = x[m.blocks[3]]

    check(
        "HTD",
        b[1:10],
        [10.9783, 7.9951, 8.7031, 8.2806, 6.3813, 3.1893, 3.3099, 3.3897, 0.6751, 0];
        atol = 1e-4,
    )
    check(
        "fixed regressions",
        b[11:15],
        [16.3082, -0.5227, -0.1245, 0.5355, -0.4195];
        atol = 2e-4,
    )
    check(
        "EBV (daily)",
        a,
        [-0.3300, -0.1604, 0.4904, 0.0043, -0.2449, -0.8367, 1.1477, 0.3786];
        atol = 1e-4,
    )
    check(
        "EBV (305 days)",
        305a,
        [-100.6476, -48.9242, 149.5718, 1.3203, -74.7065, -255.2063, 350.0481, 115.4757];
        atol = 0.02,
    )
    check("pe (daily)", pe, [-0.6156, -0.4151, -1.6853, 2.8089, -0.0928]; atol = 1e-4)

    # 5. Fitted lactation curve: v = Φ b̂₂
    # Note: the book's values from DIM 174 on are not Φ b̂₂; the verified values are checked here.
    curve = DataFrame(DIM = dims, v = Φ * b[11:15])
    println("\nFixed lactation curve:")
    show_table(curve; digits = 4)
    check(
        "lactation curve v, DIM 4–140",
        curve.v[1:5],
        [10.0835, 12.2001, 12.6254, 12.2077, 11.5679];
        atol = 2e-4,
    )

    # 6. Partition of EBV (Eqn 4.8): records corrected for HTD, curve and pe,
    #    averaged per cow (YD), and joined onto the pedigree
    rec.yc = y - [X1 X2] * b - Q * pe
    ydev = combine(groupby(rec, :cow), nrow => :n, :yc => mean => :YD)
    ped.ebv, ped.ebv305 = a, 305a
    ped.pe = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.pe[cows] = pe
    leftjoin!(ped, ydev; on = :id => :cow)
    ped.n = coalesce.(ped.n, 0)
    p = ebv_partition(ped.sire, ped.dam, σ²e / σ²a, ped.n, coalesce.(ped.YD, 0.0), a)
    ped.w_PA, ped.w_YD = p.n1, p.n2

    check("YD of cows 6 and 8", ped.YD[[6, 8]], [-0.9844, 0.3746]; atol = 1e-4)
    check("weights on YD of cows 6 and 8", p.n2[[6, 8]], [0.7882, 0.8815]; atol = 1e-4)
    check("weight on PA of cows 8 and 4", p.n1[[8, 4]], [0.1185, 0.1151]; atol = 1e-4)

    println("\nPedigree with solutions:")
    show_table(ped; digits = 4)

    (; ped, rec, curve, b, a, pe)
end
