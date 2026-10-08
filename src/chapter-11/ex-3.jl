# Examples 11.3–11.5 – GBLUP, SNP effects back-solved from GBLUP, the
# selection-index form, and cross-validation/reliability (Section 11.9).

function ex_11_3()
    println("Examples 11.3–11.5: GBLUP and equivalent models")
    (; M, dyd, ref, cand, σ²a, σ²e) = data_11()
    Z, p, k = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)                  # VanRaden (2008) method 1
    check("G = ZZ'/2Σpq", G, Z * Z' / k; atol = 1e-12)
    check("G, first column", G[:, 1], [1.472, -0.446, 0.988, 0.059, 0.685, -0.163, -0.708,
                                       -0.547, 0.887, -0.789, -0.203, -0.143, -0.829, -0.264])
    check("G, last row", G[14, :], [-0.264, 0.362, -0.466, -0.264, -0.486, -0.203, 0.100,
                                    -0.304, -0.284, 0.584, -0.526, 0.382, 0.261, 1.109])
    Gi = inv(G + 0.01I)
    W = incidence(ref, 14)
    m = mme(dyd[ref], ones(8, 1), [RandomEffect(W, Gi, σ²a)]; σ²e)
    b, (a,) = solutions(m, solve_mme(m))
    a = vec(a)
    check("GBLUP mean", b, [9.944])
    check("GBLUP DGV", a, [0.069, 0.116, 0.049, 0.260, -0.500, -0.359, 0.146, -0.231, 0.028,
                           0.115, -0.240, 0.143, 0.054, 0.353])

    # Example 11.4: SNP effects from the GBLUP solutions
    g = snp_from_gblup(Z, Gi, a; k)
    check("SNP effects back-solved from GBLUP", g,
          [0.087, -0.311, 0.262, -0.080, 0.110, 0.139, 0.000, 0.001, -0.061, -0.016])

    # Example 11.5: selection index, mean taken from GBLUP
    dgv, rel = gblup_index(G, ref, dyd[ref] .- 9.944, σ²e / σ²a)
    check("selection-index DGV", dgv, [0.070, 0.111, 0.045, 0.253, -0.495, -0.357, 0.146,
                                       -0.225, 0.028, 0.115, -0.240, 0.143, 0.054, 0.353])

    # Section 11.9: realized accuracy in the candidates, theoretical r²
    r = cor(a[cand], dyd[cand])
    check("realized accuracy, reliability", [r, r^2], [0.49, 0.24]; atol = 6e-3)
    show_table(bull = 13:26, DGV = a, theoretical_rel = rel)
    (; G, a, g, dgv, rel)
end
