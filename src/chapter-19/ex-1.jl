# Examples 19.1 and 19.2 – Jacobi (relaxation 0.8 on animal equations)
# and Gauss–Seidel iteration on the MME of Example 4.1.

function ex_19_1()
    (; m, x0) = data_19()
    println("Example 19.1: Jacobi iteration")
    ω = [1.0; 1.0; fill(0.8, 8)]
    x, it, conv, H = jacobi(m.lhs, m.rhs; x0, ω, maxiter = 20, tol = 0, history = true)
    check("round 1", H[:, 2], [4.333, 3.400, 0.267, 0.000, -0.033, -0.138, -0.411, 0.345,
                               -0.406, 0.400])
    check("round 2", H[:, 3], [4.381, 3.433, 0.164, -0.073, -0.080, -0.007, -0.248, 0.318,
                               -0.390, 0.286])
    check("round 20", H[:, 21], [4.358, 3.404, 0.099, -0.018, -0.041, -0.008, -0.185, 0.177,
                                 -0.249, 0.183])
    @printf("  CONV after 20 rounds: %.1e (book 3.0e-9)\n", conv)

    println("Example 19.2: Gauss–Seidel iteration")
    x, it, conv, H = gauss_seidel(m.lhs, m.rhs; x0, maxiter = 20, tol = 0, history = true)
    check("round 1", H[:, 2], [4.333, 3.400, 0.333, -0.083, -0.021, -0.119, -0.376, 0.392,
                               -0.364, 0.282])
    check("round 2", H[:, 3], [4.400, 3.392, 0.194, -0.035, -0.136, 0.001, -0.261, 0.254,
                               -0.284, 0.167])
    check("round 20", H[:, 21], [4.359, 3.405, 0.098, -0.019, -0.041, -0.009, -0.186, 0.177,
                                 -0.250, 0.183])
    check("= direct solution", x, solve_mme(m); atol = 1e-4)
    @printf("  CONV after 20 rounds: %.1e (book 8e-11)\n", conv)
    nothing
end
