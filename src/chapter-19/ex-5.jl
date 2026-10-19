"""
    ex_19_5() -> DataFrame

Example 19.5: Reduced Animal Model (RAM) with maternal effects solved by Gauss–Seidel
iteration (Mrode & Pocrnic, 2023, Section 19.5.3).

# Model Formulation
For the maternal RAM of [`ex_08_2`](@ref):
```math
y = Xb + Z_1 a_p + Z_2 m_p + S pe + e^*
```
where ``a_p`` and ``m_p`` are the direct and maternal breeding values of parents only
(animals 1–9). Absorbing the direct Mendelian sampling of non-parent calves (animals 10–14)
into ``e^*`` gives heterogeneous residual variances:
- For records of parent calves (animals 5–9): ``\\sigma_{e^*}^2 = \\sigma_e^2 = 350``
  (``R_{ii}^{-1} = 1/350 \\approx 0.002857``).
- For records of non-parent calves (animals 10–14):
  ``\\sigma_{e^*}^2 = \\sigma_e^2 + \\frac{1}{2} g_{11} = 350 + 75 = 425``
  (``R_{ii}^{-1} = 1/425 \\approx 0.002353``).

# Key Concepts Illustrated
1. **Gauss–Seidel on an unconstrained singular system**: Because the fixed effects (3 herds
   and 2 pens) are linearly dependent, the MME coefficient matrix is singular. Without
   imposing a constraint, Gauss–Seidel still converges to a valid solution of the normal
   equations (which specific generalized-inverse solution it reaches depends on the starting
   vector).
2. **Invariance of estimable functions and BLUPs**: Comparing the unconstrained Gauss–Seidel
   solution with the constrained solution of [`ex_08_2`](@ref) (``\\mathrm{herd}_1 = 0``)
   shows that while the individual herd and pen numbers differ, every estimable cell mean
   ``\\mathrm{herd}_i + \\mathrm{pen}_j`` and all random genetic and permanent environmental
   solutions (`direct`, `maternal`, `pe`) are identical.

# Note on Text Discrepancy
The dataset is that of Table 8.1 ([`ex_08_1`](@ref) and [`ex_08_2`](@ref)), where the second
fixed factor is called "Pen" (levels 1 and 2); in Example 19.5 the book labels those same two
levels "Sex" (Male and Female). In addition, because Mrode's relative stopping criterion
divides by ``\\|x^{(k)}\\|^2`` (which is dominated by the herd means near 30), stopping at
``10^{-14}`` leaves errors of order ``10^{-5}`` in the random effects; we iterate to
``10^{-24}`` so that the random effects and cell means match [`ex_08_2`](@ref) to 8 decimal
places.

# Returns
A `DataFrame` of the 9 parents with their `direct`, `maternal`, and maternal `pe` solutions.
"""
function ex_19_5()
    println("Example 19.5: RAM with maternal effects by Gauss–Seidel")

    # Step 1: Data and RAM design matrices
    #    Data of Example 8.1, one row per animal. The second fixed factor is
    #    "Pen" in Table 8.1 and "Sex" in Example 19.5 of the book; the data are
    #    the same.
    (; ped, G0, σ²pe, σ²e) = data_08()
    rec = dropmissing(ped, :bw)
    X = [first(incidence(rec.herd)) first(incidence(rec.pen))]
    Z1, rinv, parents, Api = ram_design(ped, rec.calf, G0[1, 1], σ²e)
    col = zeros(Int, nrow(ped))
    col[parents] = eachindex(parents)
    Z2 = incidence(col[rec.dam], length(parents))
    S, dams = incidence(rec.dam)

    # Step 2: Form RAM MME
    m = mme(
        rec.bw,
        X,
        [RandomEffect([Z1, Z2], Api, G0), RandomEffect(S, I, σ²pe)];
        Rinv = rinv,
    )
    check("1/σ²e, 1/(σ²e + ½g₁₁)", rinv[[1, 6]], [0.002857, 1 / 425]; atol = 1e-6)

    # Step 3: Gauss–Seidel iteration on the singular system
    # No constraint: Gauss–Seidel converges to a valid solution of the singular system
    # Mrode's criterion is a relative squared change: with solutions near 30, a
    # tolerance of 1e-14 still leaves absolute errors of order 1e-5, so iterate further
    x, it = gauss_seidel(m.lhs, m.rhs; tol = 1e-24, maxiter = 100_000)
    show_table(
        effect = [
            "herd " .* string.(1:3);
            "sex (pen) " .* ["M", "F"];
            "direct " .* string.(1:9);
            "maternal " .* string.(1:9);
            "pe " .* ["2", "5", "6", "7"]
        ],
        solution = x;
        digits = 3,
    )
    check("herd and sex at convergence", x[1:5], [30.563, 33.950, 31.997, 3.977, -2.872])
    check(
        "direct and maternal effects",
        x[6:23],
        [
            0.564,
            -1.246,
            1.166,
            -0.484,
            0.630,
            -0.859,
            -1.156,
            1.918,
            -0.553,
            0.261,
            -1.582,
            0.735,
            0.586,
            -0.507,
            0.841,
            1.299,
            -0.158,
            0.659,
        ];
        atol = 2e-3,
    )
    check("permanent environment", x[24:27], [-1.701, 0.415, 0.825, 0.461])
    println("  converged after ", it, " rounds")

    # Step 4: The solution of a singular system is not unique, but estimable
    #    functions are: compared with the solution with herd 1 set to zero
    #    (Example 8.2), the random effects and the cell means herdᵢ + penⱼ agree,
    #    while the individual herd and pen solutions differ
    xc = solve_mme(m; zero = [1])
    cells(v) = [v[h] + v[3+p] for h = 1:3, p = 1:2]
    check("random effects = solution with herd 1 = 0", x[6:end], xc[6:end]; atol = 1e-8)
    check("herdᵢ + penⱼ = solution with herd 1 = 0", cells(x), cells(xc); atol = 1e-8)
    @printf("  herd 1: %.3f here, 0 with the constraint\n", x[1])

    np = length(parents)
    sol = DataFrame(parent = parents, direct = x[5 .+ (1:np)], maternal = x[5+np .+ (1:np)])
    sol.pe = Vector{Union{Missing,Float64}}(missing, np)
    sol.pe[indexin(dams, parents)] = x[(end-length(dams)+1):end]
    println("\nParents: direct, maternal and maternal pe solutions:")
    show_table(sol)
    sol
end
