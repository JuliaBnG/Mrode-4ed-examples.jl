# Example 13.1 – animal model with additive and dominance effects from
# the pedigree (D by Cockerham's rule, Eqn 13.1).

function ex_13_1()
    println("Example 13.1: additive and dominance effects solved separately")
    (; ped, pig, pen, ww, σ²a, σ²d, σ²e) = data_13_1()
    N = nrow(ped)
    D = drm(ped)
    Di = inv(D)
    check("D, rows 7 and 11 (columns 7–12)", D[[7, 11], 7:12],
          [1 0 0.0625 0.0625 0.125 0.125; 0.125 0 0.125 0.125 1 0.25])
    check("D⁻¹, row 12", Di[12, 7:12], [-0.096, 0.000, -0.080, -0.080, -0.241, 1.0921];
          atol = 1e-3)
    X, _ = incidence(pen)
    Z = incidence(pig, N)
    m = mme(ww, X, [RandomEffect(Z, ainv(ped), σ²a), RandomEffect(Z, Di, σ²d)]; σ²e)
    b, (a, d) = solutions(m, solve_mme(m))
    show_table(animal = 1:N, BV = vec(a), DV = vec(d))
    check("pen", b, [16.980, 20.030])
    check("breeding values", a, [-0.160, -0.160, 0.059, 0.819, -0.320, 1.259, 0.555, -0.998,
                                 -0.350, -1.350, 1.061, -0.039])
    check("dominance deviations", d, [0, 0, 0, 0, 0.136, 0.705, 0.237, -0.993, 0, -1.333,
                                      1.428, -0.038])
    (; b, a = vec(a), d = vec(d))
end
