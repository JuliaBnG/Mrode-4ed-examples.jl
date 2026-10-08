# Example 11.13 – multivariate GBLUP (WWG and PWG of the beef calves).

function ex_11_13()
    println("Example 11.13: multivariate genomic model")
    M = [0 2 1 1 1 2 0 1 1 1
         2 1 2 1 1 0 1 0 2 0
         1 2 1 0 0 1 1 1 2 0
         1 2 2 1 0 1 1 1 2 1
         1 1 1 1 1 0 1 1 2 0]
    sex = ["M", "F", "F", "M", "M"]
    Y = [4.5 6.8; 2.9 5.0; 3.9 6.8; 3.5 6.0; 5.0 7.5]
    G0 = [20.0 18; 18 40]
    R0 = [40.0 11; 11 30]
    _, p, k = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)
    check("2Σpq", k, 4.08; atol = 0.005)
    check("G", G, [1.137 -0.725 -0.088 0.010 -0.333; -0.725 0.843 -0.235 -0.137 0.255;
                   -0.088 -0.235 0.402 0.010 -0.088; 0.010 -0.137 0.010 0.353 -0.235;
                   -0.333 0.255 -0.088 -0.235 0.402])
    X, _ = incidence(sex; levels = ["M", "F"])
    y, obs = mt_stack(Y)
    m = mme(y, reduce(hcat, mt_blocks(obs, [X, X])),
            [RandomEffect(mt_blocks(obs, [I(5), I(5)]), inv(G + 0.01I), G0)];
            Rinv = mt_rinv(R0, obs))
    b, (a,) = solutions(m, solve_mme(m))
    show_table(calf = 4:8, WWG = a[:, 1], PWG = a[:, 2])
    check("sex (WWG, PWG)", b, [4.323, 3.416, 6.753, 5.921])
    check("calves 4–8", a, [0.081 0.108; -0.212 -0.419; 0.181 0.377; -0.209 -0.302;
                            0.159 0.235])
    (; b, a)
end
