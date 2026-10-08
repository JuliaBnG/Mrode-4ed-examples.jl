# Example 6.1 – multivariate animal model, equal design matrices and no
# missing records; partition of evaluations (6.2.3) and reliabilities
# (6.2.4).

"""Univariate animal model of one trait of the beef calves, for comparison."""
function univariate(y, X, Z, Ai, σ²a, σ²e)
    m = mme(y, X, [RandomEffect(Z, Ai, σ²a)]; σ²e)
    x = solve_mme(m)
    x, m
end

function ex_06_1()
    println("Example 6.1: bivariate analysis of WWG and PWG")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    Y = [4.5 6.8; 2.9 5.0; 3.9 6.8; 3.5 6.0; 5.0 7.5]
    G0 = [20.0 18; 18 40]
    R0 = [40.0 11; 11 30]

    Ai = ainv(ped)
    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(calf, 8)
    y, obs = mt_stack(Y)
    m = mme(y, reduce(hcat, mt_blocks(obs, [X, X])),
            [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0; name = "animal")];
            Rinv = mt_rinv(R0, obs))
    x = solve_mme(m)
    b, (a,) = solutions(m, x)
    xw, _ = univariate(Y[:, 1], X, Z, Ai, 20, 40)
    xp, _ = univariate(Y[:, 2], X, Z, Ai, 40, 30)
    show_table(effect = ["male", "female", string.(1:8)...],
               WWG = x[[1, 2, 5:12...]], PWG = x[[3, 4, 13:20...]],
               WWG_uni = xw, PWG_uni = xp)
    check("sex (WWG, PWG)", b, [4.361, 3.397, 6.800, 5.880])
    check("animals, WWG", a[:, 1],
          [0.151, -0.015, -0.078, -0.010, -0.270, 0.276, -0.316, 0.244])
    check("animals, PWG", a[:, 2],
          [0.280, -0.008, -0.170, -0.013, -0.478, 0.517, -0.479, 0.392])
    check("univariate WWG", xw[3:end],
          [0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.249, 0.183])
    check("univariate PWG", [xp[1:2]; xp[3:end]],
          [6.798, 5.879, 0.277, -0.005, -0.171, -0.013, -0.471, 0.514, -0.464, 0.384])

    # 6.2.4 reliabilities r² = (gⱼⱼ − PEVᵢⱼ)/gⱼⱼ
    P = pev(m, 1)
    r² = 1 .- P ./ diag(G0)'
    show_table(animal = 1:8, PEV_WWG = P[:, 1], PEV_PWG = P[:, 2],
               r2_WWG = r²[:, 1], r2_PWG = r²[:, 2])
    check("PEV (diagonals of C⁻¹)", P,
          [18.606 35.904; 19.596 38.768; 17.893 33.799; 16.506 29.727;
           16.541 29.865; 17.152 31.504; 17.115 31.364; 16.285 29.160]; atol = 3e-3)
    check("reliabilities", r²,
          [0.070 0.102; 0.020 0.031; 0.105 0.155; 0.175 0.257;
           0.173 0.253; 0.142 0.212; 0.144 0.216; 0.186 0.271])

    # 6.2.3 â₈ = W₁PA + W₂YD
    Ri = inv(R0)
    ZRZ = [i in calf ? Ri : zeros(2, 2) for i in 1:8]
    yd = zeros(8, 2)
    yd[calf, :] = Y - [X * b[1:2] X * b[3:4]]
    p = ebv_partition_mt(ped.sire, ped.dam, G0, ZRZ, yd, a)
    check("calf 8: W₁", p.W1[8], [0.8476 -0.1191; -0.0237 0.6092]; atol = 5e-4)
    check("calf 8: PA and YD", [p.PA[8] yd[8, :]], [0.099 0.639; 0.1735 0.700])
    check("calf 8: W₁PA + W₂YD = â₈", p.W1[8] * p.PA[8] + p.W2[8] * yd[8, :], a[8, :];
          atol = 1e-10)
    (; b, a, P)
end
