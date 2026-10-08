# Example 18.2 – block Gibbs sampling for a multivariate animal model
# (WWG and PWG of Example 6.1), with inverse-Wishart updates of R and G.

function ex_18_2()
    println("Example 18.2: multivariate Gibbs sampling")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    Y = [4.5 6.8; 2.9 5.0; 3.9 6.8; 3.5 6.0; 5.0 7.5]
    G0 = [20.0 18; 18 40]
    R0 = [40.0 11; 11 30]
    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(calf, 8)
    Ai = ainv(ped)

    # BLUP of Example 6.1 for reference
    y, obs = mt_stack(Y)
    m = mme(y, reduce(hcat, mt_blocks(obs, [X, X])),
            [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0)]; Rinv = mt_rinv(R0, obs))
    b, (u,) = solutions(m, solve_mme(m))

    # (co)variances fixed: posterior means → BLUP (8 independent chains)
    chains(Yd, seed) = reduce(hcat, [gibbs_mt(Yd, X, Z, Ai; R0, G0, estimate_variances = false,
                                              niter = 50_000, burnin = 1_000,
                                              rng = Xoshiro(seed + c)).x for c in 1:8])
    mc_check("posterior means vs BLUP", chains(Y, 100), vec([reshape(b, 2, 2); u]))

    # with a missing PWG record (Example 6.2 pattern), by data augmentation
    Ym = Matrix{Union{Missing,Float64}}(Y)
    Ym[1, 2] = missing
    y2, obs2 = mt_stack(Ym)
    m2 = mme(y2, reduce(hcat, mt_blocks(obs2, [X, X])),
             [RandomEffect(mt_blocks(obs2, [Z, Z]), Ai, G0)]; Rinv = mt_rinv(R0, obs2))
    b2, (u2,) = solutions(m2, solve_mme(m2))
    mc_check("missing record: posterior means vs BLUP", chains(Ym, 200),
             vec([reshape(b2, 2, 2); u2]))

    # (co)variances sampled with proper inverse-Wishart priors centred on R0, G0
    ν = 10
    f = gibbs_mt(Y, X, Z, Ai; R0, G0, νe = ν, Ve = R0 * (ν - 3), νu = ν, Vu = G0 * (ν - 3),
                 niter = 50_000, burnin = 5_000, rng = Xoshiro(5))
    show_table(parameter = ["r11", "r21", "r22", "g11", "g21", "g22"],
               posterior_mean = vec(mean(f.samples; dims = 2)))
    (; f)
end
