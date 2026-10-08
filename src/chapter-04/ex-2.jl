# Example 4.2 – sire model.

function ex_04_2()
    println("Example 4.2: sire model")
    # records: sex of progeny, sire, WWG
    sex = ["M", "F", "F", "M", "M"]
    sire = [1, 3, 1, 4, 3]
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]
    # sire pedigree, sires recoded 1 → 1, 3 → 2, 4 → 3; sire 4 is a son of 1
    sires = [1, 3, 4]
    sped = DataFrame(sire = [0, 0, 1], dam = [0, 0, 0])
    σ²s = 0.25 * 20
    σ²e = 60 - σ²s

    X, _ = incidence(sex; levels = ["M", "F"])
    Z, _ = incidence(sire; levels = sires)
    m = mme(wwg, X, [RandomEffect(Z, ainv(sped), σ²s; name = "sire")]; σ²e)
    x = solve_mme(m)
    show_table(effect = ["male", "female", "sire " .* string.(sires)...], solution = x)
    check("A⁻¹ of sires", Matrix(ainv(sped)),
          [1.333 0 -0.667; 0 1 0; -0.667 0 1.333])
    check("MME coefficient matrix", Matrix(m.lhs),
          [3 0 1 1 1; 0 2 1 1 0; 1 1 16.666 0 -7.334; 1 1 0 13 0; 1 0 -7.334 0 15.666])
    check("solutions", x, [4.336, 3.382, 0.022, 0.014, -0.043])
    (; x)
end
