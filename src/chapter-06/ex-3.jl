# Example 6.3 – multivariate model with unequal design matrices (a
# different HYS effect for each lactation), and DYD from a multivariate
# model (6.4.2).

function ex_06_3()
    println("Example 6.3: fat yield in parities 1 and 2 as different traits")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 3, 1], dam = [0, 0, 0, 2, 2, 5, 4, 7])
    cow = 4:8
    hys1 = [1, 1, 2, 1, 2]
    hys2 = [1, 2, 1, 1, 2]
    Y = [201.0 280; 150 200; 160 190; 180 250; 285 300]
    R0 = [65.0 27; 27 70]
    G0 = [35.0 28; 28 30]

    Ai = ainv(ped)
    X1, _ = incidence(hys1)
    X2, _ = incidence(hys2)
    Z = incidence(cow, 8)
    y, obs = mt_stack(Y)
    m = mme(y, reduce(hcat, mt_blocks(obs, [X1, X2])),
            [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0)]; Rinv = mt_rinv(R0, obs))
    b, (a,) = solutions(m, solve_mme(m))
    uni = [solve_mme(mme(Y[:, t], (X1, X2)[t], [RandomEffect(Z, Ai, G0[t, t])];
                         σ²e = R0[t, t])) for t in 1:2]
    show_table(effect = ["HYS 1", "HYS 2", string.(1:8)...],
               FAT1 = [b[1:2]; a[:, 1]], FAT2 = [b[3:4]; a[:, 2]],
               FAT1_uni = uni[1], FAT2_uni = uni[2])
    check("HYS (FAT1, FAT2)", b, [175.7, 219.6, 243.2, 240.6]; atol = 0.051)
    check("animals", a,
          [8.969 8.840; -2.999 -2.777; -5.970 -6.063; 11.754 11.658; -16.253 -15.824;
           -17.314 -15.719; 8.690 8.138; 22.702 20.931])
    check("univariate FAT1 animals", uni[1][3:end],
          [6.933, -2.59, -4.341, 9.103, -12.992, -15.197, 7.566, 19.417]; atol = 5e-3)
    check("univariate FAT2 animals", uni[2][3:end],
          [8.665, -2.244, -6.422, 12.197, -15.563, -11.149, 7.696, 15.560])

    # 6.4.2 DYD of sire 1 from cows 4, 6 and 8 (one record per trait)
    Ri = inv(R0)
    ZRZ = [i in cow ? Ri : zeros(2, 2) for i in 1:8]
    yd = zeros(8, 2)
    yd[cow, :] = Y - [X1 * b[1:2] X2 * b[3:4]]
    p = ebv_partition_mt(ped.sire, ped.dam, G0, ZRZ, yd, a)
    W2prog = (Ri + 2inv(G0)) \ Ri
    check("W2prog", W2prog, [0.1713 0.0821; 0.1078 0.1244]; atol = 2e-4)
    # The book prints YD₄₁ = 25.7 although 201 − 175.7 = 25.3; with that slip
    # corrected its own numbers give DYD₁ = (24.254, 32.154) instead of
    # (24.521, 32.154), and M₃DYD = (7.388, 7.358) instead of (7.439, 7.387).
    check("DYD of sire 1 (book's slip corrected)", p.DYD[1], [24.254, 32.154]; atol = 0.05)
    check("M₃ of sire 1", p.M3[1], [0.1937 0.0836; 0.1099 0.1459]; atol = 3e-4)
    check("M₃ DYD (no granddaughter information)", p.M3[1] * p.DYD[1], [7.388, 7.358];
          atol = 0.02)
    (; b, a, partition = p)
end
