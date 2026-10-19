"""
    ex_19_5() -> DataFrame

Run Example 19.5 from Mrode & Pocrnic (2023): Solving the Reduced Animal Model (RAM)
with maternal effects (from Example 8.2) using Gauss–Seidel iteration.

# Model Formulation
For the maternal RAM:
```math
\\mathbf{y} = X\\mathbf{b} + Z_1 \\mathbf{a}_{p} + Z_2 \\mathbf{m}_{p} + S\\mathbf{pe} +
    \\mathbf{e}^*
```
where ``\\mathbf{a}_p`` and ``\\mathbf{m}_p`` are direct and maternal additive genetic
merits of parents only. Non-parent progeny are absorbed into the residual error, modifying
the effective residual variance:
-  For records of parent calves: ``\\sigma_{e^*}^2 = \\sigma_e^2 = 350``
  (``r^{ii} = 1/350 \\approx 0.002857``)
-  For records of non-parent calves:
  ``\\sigma_{e^*}^2 = \\sigma_e^2 + \\frac{1}{2} g_{11} = 350 + 75 = 425``
  (``r^{ii} = 1/425``)

# Gauss–Seidel on Singular Systems
Because the fixed effects (3 herds, 2 pens) are linearly dependent, the MME coefficient
matrix is singular. Gauss–Seidel iteration converges to one of the solutions (which one
depends on the starting values) without requiring explicit full-rank reparameterization.

# Estimable Functions
The Gauss–Seidel solution is compared with the solution constrained by herd 1 = 0
(Example 8.2): the random effects and the cell means ``\\mathrm{herd}_i + \\mathrm{pen}_j``
agree, while the individual herd and pen solutions do not.

# Returns
The table of parents with their direct, maternal and maternal pe solutions; the
solutions are printed and checked against the book.
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
