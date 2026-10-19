"""
    ex_19_1() -> DataFrame

Examples 19.1 & 19.2: Solving the Mixed Model Equations by Jacobi and Gauss–Seidel iteration
(Mrode & Pocrnic, 2023, Section 19.4).

# Model Formulation
For the linear system ``Cx = r`` (here the 10-equation univariate animal model MME of
[`ex_04_1`](@ref)):
- **Jacobi iteration with relaxation factor ``\\omega_i``** (Example 19.1):
  ```math
  x_i^{(k+1)} = x_i^{(k)} + \\frac{\\omega_i}{C_{ii}} \\left( r_i - \\sum_{j=1}^n
      C_{ij} x_j^{(k)} \\right)
  ```
  Every equation is updated using only the previous round's vector ``x^{(k)}``. An
  under-relaxation weight ``\\omega_i = 0.8`` is applied to the animal equations (with
  ``\\omega_i = 1.0`` on the fixed sex equations) so that the iteration converges.
- **Gauss–Seidel iteration** (Example 19.2):
  ```math
  x_i^{(k+1)} = \\frac{1}{C_{ii}} \\left( r_i - \\sum_{j < i} C_{ij} x_j^{(k+1)} - \\sum_{j
      > i} C_{ij} x_j^{(k)} \\right)
  ```
  Each update immediately uses the newly computed entries ``x_j^{(k+1)}`` (``j < i``) within
  the same sweep, converging faster without needing under-relaxation.

# Key Concepts Illustrated
1. **Computing Mrode's starting values from the data table**: In `data_19`, the
   starting vector ``x^{(0)}`` is computed directly from the `DataFrame` of records using
   `groupby` (raw sex means for fixed effects, deviations from the sex mean for recorded
   calves, and 0 for unrecorded ancestors).
2. **Both iterative schemes written out as explicit loops**: We write out the 20 rounds of
   relaxed Jacobi and Gauss–Seidel in plain Julia, verify that they match `jacobi` and
   `gauss_seidel`, and tabulate Mrode's relative squared-change convergence criterion
   ``\\|x^{(k)} - x^{(k-1)}\\|^2 / \\|x^{(k)}\\|^2`` side by side by round.

# Returns
A `DataFrame` with columns `round`, `conv_jacobi`, and `conv_gauss_seidel` over the 20
iterations.
"""
function ex_19_1()
    (; m, x0) = data_19()
    check(
        "starting values from the records",
        x0,
        [4.333, 3.400, 0, 0, 0, 0.167, -0.500, 0.500, -0.833, 0.667],
    )
    C, r = Matrix(m.lhs), m.rhs
    n = length(r)
    conv(x, xo) = sum(abs2, x - xo) / sum(abs2, x)  # Mrode's convergence criterion

    # Step 1: Jacobi iteration with relaxation factor ω = 0.8 on animal equations
    println("Example 19.1: Jacobi iteration")
    ω = [1.0; 1.0; fill(0.8, 8)]
    x, it, cj, H = jacobi(m.lhs, m.rhs; x0, ω, maxiter = 20, tol = 0, history = true)
    #    written out: every equation uses the previous round only,
    #    xᵢ ← xᵢ + ωᵢ (rᵢ - Cᵢ'x)/Cᵢᵢ
    xj = copy(x0)
    convj = Float64[]
    for _ = 1:20
        xo = copy(xj)
        xj = xo + ω .* (r - C * xo) ./ diag(C)
        push!(convj, conv(xj, xo))
    end
    check("Jacobi written out = jacobi() (round 20)", xj, H[:, 21]; atol = 1e-12)
    check(
        "round 1",
        H[:, 2],
        [4.333, 3.400, 0.267, 0.000, -0.033, -0.138, -0.411, 0.345, -0.406, 0.400],
    )
    check(
        "round 2",
        H[:, 3],
        [4.381, 3.433, 0.164, -0.073, -0.080, -0.007, -0.248, 0.318, -0.390, 0.286],
    )
    check(
        "round 20",
        H[:, 21],
        [4.358, 3.404, 0.099, -0.018, -0.041, -0.008, -0.185, 0.177, -0.249, 0.183],
    )
    @printf("  CONV after 20 rounds: %.1e (book 3.0e-9)\n", cj)

    # Step 2: Gauss–Seidel iteration
    println("Example 19.2: Gauss–Seidel iteration")
    x, it, cg, H = gauss_seidel(m.lhs, m.rhs; x0, maxiter = 20, tol = 0, history = true)
    #    written out: each equation uses the values already updated in this round
    xg = copy(x0)
    convg = Float64[]
    for _ = 1:20
        xo = copy(xg)
        for i = 1:n
            xg[i] += (r[i] - C[i, :]' * xg) / C[i, i]
        end
        push!(convg, conv(xg, xo))
    end
    check("Gauss–Seidel written out (round 20)", xg, H[:, 21]; atol = 1e-12)
    check(
        "round 1",
        H[:, 2],
        [4.333, 3.400, 0.333, -0.083, -0.021, -0.119, -0.376, 0.392, -0.364, 0.282],
    )
    check(
        "round 2",
        H[:, 3],
        [4.400, 3.392, 0.194, -0.035, -0.136, 0.001, -0.261, 0.254, -0.284, 0.167],
    )
    check(
        "round 20",
        H[:, 21],
        [4.359, 3.405, 0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.250, 0.183],
    )
    check("= direct solution", x, solve_mme(m); atol = 1e-4)
    @printf("  CONV after 20 rounds: %.1e (book 8e-11)\n", cg)

    rounds = DataFrame(round = 1:20, conv_jacobi = convj, conv_gauss_seidel = convg)
    println("\nConvergence criterion by round:")
    fmt = ByRow(v -> @sprintf("%.1e", v))
    println(
        transform(rounds, [:conv_jacobi, :conv_gauss_seidel] .=> fmt; renamecols = false),
    )
    rounds
end
