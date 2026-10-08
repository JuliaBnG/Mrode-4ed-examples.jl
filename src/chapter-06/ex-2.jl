# Example 6.2 – multivariate animal model, equal design matrices with
# missing records.

function ex_06_2()
    println("Example 6.2: bivariate analysis with missing PWG records")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3, 7], dam = [0, 0, 0, 0, 2, 2, 5, 6, 0])
    calf = 4:9
    sex = ["M", "F", "F", "M", "M", "F"]
    Y = [4.5 missing; 2.9 5.0; 3.9 6.8; 3.5 6.0; 5.0 7.5; 4.0 missing]
    G0 = [20.0 18; 18 40]
    R0 = [40.0 11; 11 30]

    Ai = ainv(ped)
    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(calf, 9)
    y, obs = mt_stack(Y)
    Ri = mt_rinv(R0, obs)       # inverses of R_o and R_m by missing pattern
    m = mme(y, reduce(hcat, mt_blocks(obs, [X, X])),
            [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0)]; Rinv = Ri)
    b, (a,) = solutions(m, solve_mme(m))
    check("X₁'R¹¹X₁", Matrix(m.lhs[1:2, 1:2]), [0.081 0; 0 0.081])

    # univariate analyses use only the observed records of each trait
    uni = map(1:2) do t
        r = findall(obs[:, t])
        mu = mme(Float64.(Y[r, t]), X[r, :], [RandomEffect(Z[r, :], Ai, G0[t, t])];
                 σ²e = R0[t, t])
        solve_mme(mu)
    end
    show_table(effect = ["male", "female", string.(1:9)...],
               WWG = [b[1:2]; a[:, 1]], PWG = [b[3:4]; a[:, 2]],
               WWG_uni = uni[1], PWG_uni = uni[2])
    check("sex (WWG, PWG)", b, [4.367, 3.657, 6.834, 6.007])
    check("animals", a,
          [0.130 0.266; -0.084 -0.075; -0.098 -0.194; 0.007 0.016; -0.343 -0.555;
           0.192 0.440; -0.308 -0.483; 0.201 0.349; -0.018 -0.119])
    check("univariate WWG", uni[1],
          [4.364, 3.648, 0.077, -0.081, -0.058, 0.003, -0.250, 0.098, -0.237, 0.143, 0.010])
    check("univariate PWG", uni[2],
          [6.784, 5.873, 0.273, 0.000, -0.165, -0.025, -0.463, 0.517, -0.460, 0.392, -0.230])
    (; b, a)
end
