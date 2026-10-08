# Example 4.3 – reduced animal model, with back-solving for non-parents.

function ex_04_3()
    println("Example 4.3: reduced animal model")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = [4, 5, 6, 7, 8]
    sex = ["M", "F", "F", "M", "M"]
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]
    σ²a, σ²e = 20.0, 40.0

    W, rinv, parents, Ap⁻¹ = ram_design(ped, calf, σ²a, σ²e)   # Ap⁻¹: parents only
    X, _ = incidence(sex; levels = ["M", "F"])
    m = mme(wwg, X, [RandomEffect(W, Ap⁻¹, σ²a; name = "parents")]; Rinv = rinv)
    x = solve_mme(m)
    b, (ap,) = solutions(m, x)
    check("R⁻¹", rinv, [0.025, 0.025, 0.025, 0.020, 0.020])
    check("W'R⁻¹y", (W' * (rinv .* wwg)), [0, 0, 0.050, 0.148, 0.107, 0.148])
    check("sex and parent solutions", x,
          [4.358, 3.404, 0.098, -0.019, -0.041, -0.009, -0.186, 0.177])
    check("non-zeros in the coefficient matrix", nnz(m.lhs), 38)

    a = ram_backsolve(ped, calf, wwg - X * b, vec(ap), parents, σ²a, σ²e)
    show_table(animal = 1:8, ebv = a)
    check("back-solved non-parents 7 and 8", a[7:8], [-0.249, 0.183])
    (; b, a)
end
