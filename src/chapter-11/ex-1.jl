# Example 11.1 – SNPs fitted as fixed effects together with a random
# polygenic (animal) effect, without and with EDC weights.

function ex_11_1()
    println("Example 11.1: three SNPs as fixed effects plus polygenic effects")
    (; ped, M, dyd, edc, id, ref, cand, σ²a, σ²e) = data_11()
    Z, p, _ = center_genotypes(M)
    check("allele frequencies", p, [0.321, 0.179, 0.357, 0.357, 0.143, 0.607, 0.071, 0.964,
                                    0.571, 0.393])
    Z3 = Z[:, 1:3]
    y = dyd[ref]
    X = [ones(8) Z3[ref, :]]
    W = incidence(id[ref], nrow(ped))
    Ai = ainv(ped)
    res = map([(; σ²e), (; Rinv = edc[ref] ./ σ²e)]) do r
        m = mme(y, X, [RandomEffect(W, Ai, σ²a)]; r...)
        b, (u,) = solutions(m, solve_mme(m))
        (; b, u = vec(u), dgv = Z3 * b[2:4])
    end
    for (lab, r) in zip(("unweighted", "weighted"), res)
        show_table(bull = id, DGV = r.dgv, polygenic = r.u[id])
    end
    uw, wt = res
    check("unweighted: mean and SNP effects", uw.b, [9.895, 0.607, -4.080, 1.934])
    check("weighted: mean and SNP effects", wt.b, [9.196, 1.158, -3.956, 2.527])
    check("unweighted: DGV", uw.dgv, [2.834, 0.293, 0.081, 3.554, -2.460, -3.787, 1.620,
                                      -2.460, 0.900, -0.314, -2.460, 0.293, -0.314, 2.227])
    check("unweighted: polygenic", uw.u[id], [-0.299, 0.256, 0.142, 0.254, -0.085, 0.271,
                                              -0.092, -0.181, 0, 0.128, 0.128, 0.128, 0.128,
                                              0.128])
    check("weighted: DGV", wt.dgv, [3.708, 0.022, 1.120, 3.918, -2.565, -3.934, 1.391,
                                    -2.565, 1.181, -1.136, -2.565, 0.022, -1.136, 2.549];
          atol = 3e-3)  # large EDC weights magnify the book's rounding (α = 6.952)
    check("weighted: polygenic", wt.u[id], [-3.821, 4.116, 2.248, 2.158, -0.450, 2.402,
                                            -0.212, -1.542, 0, 2.058, 2.058, 2.058, 2.058,
                                            2.058]; atol = 3e-3)
    check("GEBV of bull 13 (unweighted)", uw.dgv[1] + uw.u[13], 2.535)
    res
end
