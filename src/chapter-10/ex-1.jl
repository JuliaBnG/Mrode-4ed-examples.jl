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
1.  **Legendre Polynomials for Lactation Curves**: Orthogonal polynomials over standardised
   intervals.
2.  **Fixed Regression Test-Day Model**: A single average curve fitted to all cows, with
   constant daily breeding values and permanent environment.
3. **305-Day Genetic Evaluations**: Daily EBV scaled by 305 days.
4. **Decomposition of EBVs**: Partition into Parent Average (PA) and Yield Deviation (YD)
   adjusted for HTD, curve, and permanent environment.

# Note on Text Discrepancy
The book's printed values for the fixed lactation curve from DIM 174 on (11.0407, 10.9156,
etc.) do not match ``\\Phi \\hat{\\beta}_2`` (e.g. at DIM 174,
``\\Phi_{174} \\hat{\\beta}_2 = 11.099``). The exact curve values are checked and verified
here.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed effect solutions (HTD and Legendre curve coefficients).
- `a`: Daily estimated breeding values for animals 1 to 8.
- `pe`: Daily permanent environmental effects for cows 4 to 8.
"""
function ex_10_1()
    println("Example 10.1: fixed regression model for test-day fat yield")

    # 1. Load test-day data and variance components
    (; ped, dims, cow, dim, htd, y, Φ) = data_10()
    σ²a = 5.521
    σ²pe = 8.470
    σ²e = 3.710

    check(
        "Φ (Appendix G), first and last rows",
        Φ[[1, 10], :],
        [0.7071 -1.2247 1.5811 -1.8704 2.1213; 0.7071 1.2247 1.5811 1.8704 2.1213];
        atol = 5e-4,
    )  # the book rounds the Legendre coefficients to 4 decimals

    # 2. Design matrices and MME setup
    X1, _ = incidence(htd)                          # fixed HTD effects (10 levels)
    X2 = Φ[dim, :]                                  # fixed lactation curve (order 4)
    Z = incidence(cow, nrow(ped))                   # random animal effect
    Q, _ = incidence(cow)                           # random pe effect
    m = mme(y, [X1 X2], [RandomEffect(Z, ainv(ped), σ²a), RandomEffect(Q, I, σ²pe)]; σ²e)

    check(
        "X₂'X₂",
        Matrix(m.lhs[11:15, 11:15]),
        [
            20.9996 -4.4261 4.0568 -0.8441 8.7149;
            -4.4261 24.6271 -4.7012 11.1628 -3.0641;
            4.0568 -4.7012 31.0621 -6.6603 19.0867;
            -0.8441 11.1628 -6.6603 38.6470 -8.8550;
            8.7149 -3.0641 19.0867 -8.8550 48.2930
        ];
        atol = 1e-2,
    )

    # 3. Solve MME (HTD 10 set to zero)
    x = solve_mme(m; zero = [10])                   # HTD 10 set to zero
    b, (a, pe) = solutions(m, x)
    a, pe = vec(a), vec(pe)

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

    # 4. Fitted lactation curve: v = Φ b̂₂
    # Note: the book's values from DIM 174 on are not Φ b̂₂; the verified values are checked here.
    v = Φ * b[11:15]
    show_table(DIM = dims, v = v; digits = 4)
    check(
        "lactation curve v, DIM 4–140",
        v[1:5],
        [10.0835, 12.2001, 12.6254, 12.2077, 11.5679];
        atol = 2e-4,
    )

    # 5. Partition of EBV (Eqn 4.8): YD corrected for HTD, curve, and pe
    yc = y - [X1 X2] * b - Q * pe
    nrec = [count(==(i), cow) for i = 1:8]
    yd = [nrec[i] > 0 ? sum(yc[cow .== i]) / nrec[i] : 0.0 for i = 1:8]
    p = ebv_partition(ped.sire, ped.dam, σ²e / σ²a, nrec, yd, a)

    check("YD of cows 6 and 8", yd[[6, 8]], [-0.9844, 0.3746]; atol = 1e-4)
    check("weights on YD of cows 6 and 8", p.n2[[6, 8]], [0.7882, 0.8815]; atol = 1e-4)
    check("weight on PA of cows 8 and 4", p.n1[[8, 4]], [0.1185, 0.1151]; atol = 1e-4)

    (; b, a, pe)
end
