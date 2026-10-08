# Example 19.5 – reduced animal model with maternal effects (Example 8.2)
# solved by Gauss–Seidel iteration.

function ex_19_5()
    println("Example 19.5: RAM with maternal effects by Gauss–Seidel")
    (; ped, calf, herd, pen, bw, G0, σ²pe, σ²e) = data_08()
    dam = ped.dam[calf]
    X = [first(incidence(herd)) first(incidence(pen))]
    Z1, rinv, parents, Api = ram_design(ped, calf, G0[1, 1], σ²e)
    col = zeros(Int, nrow(ped))
    col[parents] = eachindex(parents)
    Z2 = incidence(col[dam], length(parents))
    S, _ = incidence(dam)
    m = mme(bw, X, [RandomEffect([Z1, Z2], Api, G0),
                    RandomEffect(S, I, σ²pe)]; Rinv = rinv)
    check("1/σ²e, 1/(σ²e + ½g₁₁)", rinv[[1, 6]], [0.002857, 1 / 425]; atol = 1e-6)
    # no constraint: Gauss–Seidel converges to a solution of the singular system
    x, it = gauss_seidel(m.lhs, m.rhs; tol = 1e-14, maxiter = 100_000)
    show_table(effect = ["herd " .* string.(1:3); "sex " .* ["M", "F"];
                         "direct " .* string.(1:9); "maternal " .* string.(1:9);
                         "pe " .* ["2", "5", "6", "7"]], solution = x; digits = 3)
    check("herd and sex at convergence", x[1:5], [30.563, 33.950, 31.997, 3.977, -2.872])
    check("direct and maternal effects", x[6:23],
          [0.564, -1.246, 1.166, -0.484, 0.630, -0.859, -1.156, 1.918, -0.553,
           0.261, -1.582, 0.735, 0.586, -0.507, 0.841, 1.299, -0.158, 0.659]; atol = 2e-3)
    check("permanent environment", x[24:27], [-1.701, 0.415, 0.825, 0.461])
    println("  converged after ", it, " rounds")
    nothing
end
