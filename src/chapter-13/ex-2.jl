# Example 13.2 – solving directly for total genetic merit g = a + d with
# var(g) = Aσ²a + Dσ²d, and recovering a and d from g.

function ex_13_2()
    println("Example 13.2: total genetic merit")
    (; ped, pig, pen, ww, σ²a, σ²d, σ²e) = data_13_1()
    A, D = nrm(ped), drm(ped)
    V = σ²a * A + σ²d * D
    Vi = inv(V)
    X, _ = incidence(pen)
    m = mme(ww, X, [RandomEffect(incidence(pig, nrow(ped)), Vi, 1.0)]; σ²e)
    b, (g,) = solutions(m, solve_mme(m))
    g = vec(g)
    check("pen", b, [16.980, 20.030])
    check("total genetic merit", g, [-0.160, -0.160, 0.059, 0.819, -0.184, 1.963, 0.792,
                                     -1.991, -0.349, -2.683, 2.489, -0.078])
    a = σ²a * A * (Vi * g)
    d = σ²d * D * (Vi * g)
    r = ex_13_1()
    check("â = σ²a A M⁻¹ g (Example 13.1)", a, r.a; atol = 1e-8)
    check("d̂ = σ²d D M⁻¹ g (Example 13.1)", d, r.d; atol = 1e-8)
    (; b, g)
end
