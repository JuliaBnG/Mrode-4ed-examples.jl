# Example 4.4 – animal model with unknown parent groups (QP transformation).

function ex_04_4()
    println("Example 4.4: animal model with groups")
    # unknown sires → group 1 (animal 9), unknown dams → group 2 (animal 10)
    ped = DataFrame(sire = [-1, -1, -1, 1, 3, 1, 4, 3], dam = [-2, -2, -2, -2, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]
    σ²a, σ²e = 20.0, 40.0

    Ai = ainv_upg(ped)
    check("A⁻¹ rows of the groups", Matrix(Ai)[9:10, :],
          [-0.5 -0.5 -0.5 0 0 0 0 0 0.75 0.75; -0.17 -0.5 -0.5 -0.67 0 0 0 0 0.75 1.08];
          atol = 5e-3)
    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(calf, size(Ai, 1))  # no records on groups
    m = mme(wwg, X, [RandomEffect(Z, Ai, σ²a; name = "animal+group")]; σ²e)
    x = solve_mme(m; zero = [m.blocks[2][9]])  # group 1 set to zero
    show_table(effect = ["male", "female", string.(1:8)..., "G9", "G10"], solution = x)
    check("solutions (G9 = 0)", x,
          [5.458, 4.313, -0.767, -0.923, -0.963, -1.268, -1.099, -0.728, -1.338, -0.768,
           0, -1.769])

    # â* = â + Qĝ, with â from the model without groups (Example 4.1)
    Q = group_contributions(ped)
    check("Q′ (= (TQ*)′)", Q',
          [0.5 0.5 0.5 0.25 0.5 0.5 0.375 0.5; 0.5 0.5 0.5 0.75 0.5 0.5 0.625 0.5]; atol = 1e-12)
    ped0 = DataFrame(sire = max.(ped.sire, 0), dam = max.(ped.dam, 0))
    m0 = mme(wwg, X, [RandomEffect(incidence(calf, 8), ainv(ped0), σ²a)]; σ²e)
    a0 = solve_mme(m0)[3:end]
    astar = a0 + Q * x[end-1:end]
    show_table(animal = 1:8, a_groups = x[3:10], a_nogroup_plus_Qg = astar)

    # the same model without the QP transformation: y = Xb + ZQg + Za + e
    mq = mme(wwg, [X Z[:, 1:8] * Q], [RandomEffect(incidence(calf, 8), ainv(ped0), σ²a)]; σ²e)
    xq = solve_mme(mq; zero = [3])
    check("â + Qĝ from the explicit model = QP solutions",
          xq[mq.blocks[2]] + Q * xq[3:4], x[3:10]; atol = 1e-8)
    (; x, astar)
end
