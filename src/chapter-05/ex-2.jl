# Example 5.2 – animal model with common environmental (full-sib) effects.

function ex_05_2()
    println("Example 5.2: common environmental effects, piglet weaning weight")
    ped = DataFrame(sire = [0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1],
                    dam = [0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5])
    piglet = 6:15
    sex = ["M", "F", "F", "F", "M", "F", "F", "M", "F", "M"]
    ww = [90, 70, 65, 98, 106, 60, 80, 100, 85, 68.0]
    σ²a, σ²c, σ²e = 20.0, 15.0, 65.0

    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(piglet, nrow(ped))
    W, dams = incidence(ped.dam[piglet])      # litters of dams 2, 4, 5
    m = mme(ww, X, [RandomEffect(Z, ainv(ped), σ²a; name = "animal"),
                    RandomEffect(W, I, σ²c; name = "litter")]; σ²e)
    x = solve_mme(m)
    b, (a, c) = solutions(m, x)
    show_table(effect = ["male"; "female"; string.(1:15); "litter of " .* string.(dams)],
               solution = x)
    check("sex", b, [91.493, 75.764])
    check("animals", a, [-1.441, -1.175, 1.441, 1.441, -0.266, -1.098, -1.667, -2.334,
                         3.925, 2.895, -1.141, 1.525, 0.448, 0.545, -3.819])
    check("common environment", c, [-1.762, 2.161, -0.399])
    (; b, a, c)
end
