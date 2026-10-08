# Example 12.2 – inverse of G by the APY algorithm (core: bulls 15, 20,
# 22 and 26) and GBLUP with it.

function ex_12_2()
    println("Example 12.2: APY inverse of G")
    (; M, dyd, id, ref, σ²a, σ²e) = data_11()
    _, p, _ = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)
    core = findall(in([15, 20, 22, 26]), id)
    nc = setdiff(1:14, core)
    Gi = apy_ginv(G, core)
    Mnn = 1 ./ diag(Gi)[nc]
    check("diag(Mₙₙ)", Mnn, [0.234, 0.204, 0.600, 0.848, 0.485, 0.420, 1.107, 0.104, 0.644,
                            0.102])
    # the book prints G⁻¹_APY with the core bulls first (15, 20, 22, 26, 13, …)
    o = [core; nc]
    check("G⁻¹_APY: first row (bull 15, APY order)", Matrix(Gi)[o[1], o],
          [4.425, 0.373, -0.527, 0.412, -0.795, 2.731, -0.349, -0.658, 0.959, 0.776, 0.483,
           -0.447, -0.108, 1.828])
    check("non-core block is diagonal", nnz(Gi[nc, nc]), length(nc); atol = 0)
    W = incidence(ref, 14)
    dgv(K) = solve_mme(mme(dyd[ref], ones(8, 1), [RandomEffect(W, K, σ²a)]; σ²e))[2:end]
    a_apy = dgv(Gi)
    a_full = dgv(inv(G + 0.01I))
    show_table(bull = id, full = a_full, APY = a_apy)
    # the book omits the minus sign of bull 17 (−0.471)
    check("DGV with G⁻¹_APY", a_apy, [0.065, 0.119, 0.030, 0.365, -0.471, -0.244, -0.055,
                                      -0.220, 0.010, 0.127, -0.229, 0.097, -0.007, 0.363])
    check("cor(full, APY)", cor(a_full, a_apy), 0.95; atol = 0.005)
    (; Gi, a_apy, a_full)
end
