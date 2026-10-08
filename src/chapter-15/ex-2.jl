# Example 15.2 – joint analysis of birth weight (linear) and calving
# difficulty (binary threshold) (Foulley et al., 1983), and the
# bivariate linear model.

function ex_15_2()
    println("Example 15.2: birth weight and calving difficulty")
    bw = [41, 37.5, 41.5, 40, 43, 42, 35, 46, 40.5, 39, 41.4, 43, 34, 47, 42, 44.5, 49, 41.6,
          36, 42.7, 32.5, 44.4, 46, 47, 51, 39, 44.5, 40.5, 43.5, 42.5, 48.8, 38.5, 52, 48, 41,
          50.5, 43.7, 51, 51.6, 45.3, 36.5, 50.5, 46, 45, 36, 43.5, 36.5]
    cd = [zeros(11); 1; 0; 1; zeros(9); 1; 1; zeros(5); 1; 0; 0; 0; 0; ones(5); 0; 0; 1; 0;
          0; 0; 0]
    sex = [1, 1, fill(2, 8)..., 1, 1, 2, fill(1, 6)..., 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1,
           1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2]
    origin = [fill(1, 7); fill(2, 3); fill(1, 5); 2; 2; 1; fill(2, 5); 1; 1; 1; 2; fill(1, 7);
              2; 2; 2; 2; fill(1, 7); 2; 2]
    season = [1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1,
              1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1]
    sire = [fill(1, 10); fill(2, 7); fill(3, 6); fill(4, 4); fill(5, 11); fill(6, 9)]
    # sire – maternal grandsire pedigree of the bulls
    Ai = Matrix(ainv_smgs(DataFrame(sire = [0, 0, 1, 2, 3, 2], mgs = [0, 0, 0, 1, 2, 3])))
    check("A⁻¹ (sire–MGS), first row", Ai[1, :], [1.424, 0.182, -0.667, -0.364, 0, 0])
    X = [first(incidence(origin)) first(incidence(season)) first(incidence(sex))]
    Z, _ = incidence(sire)
    G = [0.7178 0.1131; 0.1131 0.0466]
    r = joint_binary_quantitative(bw, cd, X, Z, X, Z, Ai, G, 20.0, 0.459; ycentre = 43.02,
                                  zero1 = [4, 6], zero2 = [4, 6])
    check("residual regression β", r.β, 0.1155)
    names = ["origin 1", "origin 2", "season 1", "season 2", "male", "female",
             "sire " .* string.(1:6)...]
    show_table(effect = names, BW = [r.b1; r.u1], CD = [r.τ; r.ν]; digits = 4)
    check("iteration 0: BW", r.history[1:12, 1],
          [41.6177, 42.2069, -1.2359, 0, 3.1728, 0, -0.3497, 0.1201, -0.2852, 0.2022, 0.2994,
           0.1794]; atol = 1e-3)
    check("iteration 0: CD", r.history[13:24, 1],
          [0.1343, 0.0851, -0.0317, 0, 0.2437, 0, -0.0450, 0.0237, -0.0363, 0.0289, 0.0188,
           0.0272]; atol = 1e-3)
    check("converged: BW", [r.b1; r.u1],
          [41.6192, 42.2109, -1.2344, 0, 3.1690, 0, -0.3592, 0.1303, -0.2948, 0.2126, 0.2969,
           0.1815]; atol = 1e-3)
    check("converged: CD", [r.τ; r.ν],
          [-1.3936, -1.7457, 0.1404, 0, 0.8401, 0, -0.0573, 0.0372, -0.0485, 0.0411, 0.0152,
           0.0309]; atol = 1e-3)
    println("  converged after ", r.iterations, " iterations")
    # probability of difficult calving: heifer of origin 2, season 1, male calf
    V = [cdf(Normal(), r.τ[2] + r.τ[3] + r.τ[5] + r.ν[j] +
                       r.β * (r.b1[2] + r.b1[3] + r.b1[5] - 43.02)) for j in 1:6]
    check("P(difficult | origin 2, season 1, male)", V,
          [0.245, 0.275, 0.247, 0.276, 0.268, 0.273]; atol = 2e-3)

    # bivariate linear model (the R analysis of Section 15.3)
    R0 = [20 2.089; 2.089 1.036]
    y, obs = mt_stack([bw cd])
    keepX = setdiff(1:6, [4, 6])
    m = mme(y, reduce(hcat, mt_blocks(obs, [X[:, keepX], X[:, keepX]])),
            [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G)]; Rinv = mt_rinv(R0, obs))
    b, (u,) = solutions(m, solve_mme(m))
    check("linear model: BW", [b[1:4]; u[:, 1]],
          [41.6175, 42.2022, -1.2387, 3.1845, -0.3268, 0.1171, -0.2641, 0.1886, 0.2688,
           0.1690]; atol = 1e-3)
    check("linear model: CD", [b[5:8]; u[:, 2]],
          [0.1349, 0.0876, -0.0311, 0.2410, -0.0527, 0.0285, -0.0427, 0.0350, 0.0195, 0.0323];
          atol = 1e-3)
    r
end
