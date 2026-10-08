# Example 12.1 – single-step GBLUP: G blended with A₂₂, tuned to the
# pedigree base, and H⁻¹ = A⁻¹ + [0 0; 0 Gt⁻¹ − A₂₂⁻¹].

function ex_12_1()
    println("Example 12.1: single-step GBLUP")
    (; ped, M, dyd, id, ref, σ²a, σ²e) = data_11()
    _, p, _ = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)
    A22 = nrm(ped, id)                            # Colleau's algorithm
    Gb = 0.95G + 0.05A22                          # blending
    Gt, a, b = tune_grm(Gb, A22)                  # tuning (Chen et al., 2011)
    check("tuning: Gt = a + b Gb", [a, b], [0.202, 0.709])
    Hi = hinv(ped, Gt, id)
    W = incidence(id[ref], nrow(ped))
    m = mme(dyd[ref], ones(8, 1), [RandomEffect(W, Hi, σ²a)]; σ²e)
    x = solve_mme(m)
    show_table(animal = 1:26, ssGBLUP = x[2:end])
    check("ssGBLUP, ungenotyped 1–12", x[2:13],
          [0.009, 0.121, 0.009, 0.010, -0.250, -0.196, -0.001, 0.026, -0.076, 0.040,
           -0.137, 0.136])
    check("ssGBLUP, genotyped 13–26", x[14:27],
          [0.048, 0.082, 0.039, 0.202, -0.356, -0.254, 0.098, -0.169, 0.018, 0.079, -0.164,
           0.101, 0.039, 0.245])
    (; Gt, Hi, x)
end
