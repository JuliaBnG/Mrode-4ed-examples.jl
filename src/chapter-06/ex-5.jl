# Example 6.5 – multi-trait across-country evaluation (MACE) of bulls
# with a sire–maternal-grandsire model and phantom groups, and the
# partition of MACE proofs (Eqns 6.19–6.21).

function ex_06_5()
    println("Example 6.5: MACE for two countries")
    # bulls 1–9; unknown ancestors in groups G1–G5 (coded −1 … −5)
    ped = DataFrame(sire = [7, 8, 7, 1, 8, 1, -1, -1, -1],
                    mgs = [-3, 9, 2, -2, -3, 9, -2, -2, -3],
                    mgd = [-5, -5, -5, -5, -4, -4, -4, -4, -4])
    bull = [[1, 2, 3, 4], [1, 2, 5, 6]]
    edc = [[58, 150, 20, 25], [90, 65, 30, 55]]
    drp = [[9.7229, 9.9717, 19.2651, -8.5711], [14.5088, 7.7594, 23.9672, -9.6226]]
    σ²e = [206.5, 148.5]
    G0 = [20.5 12.839; 12.839 9.5]

    Ai = ainv_smgs(ped)
    q = size(Ai, 1)
    y = reduce(vcat, drp)
    rinv = reduce(vcat, [edc[c] ./ σ²e[c] for c in 1:2])
    n1, n2 = length.(bull)
    X = [ones(n1) zeros(n1); zeros(n2) ones(n2)]
    Z1 = [incidence(bull[1], q); spzeros(n2, q)]
    Z2 = [spzeros(n1, q); incidence(bull[2], q)]
    m = mme(y, X, [RandomEffect([Z1, Z2], Ai, G0; name = "bull+group")]; Rinv = rinv)
    # 1 is added to the diagonals of the group equations to remove their
    # dependencies, i.e. groups are treated as random with unit variance
    for k in 1:2, g in 10:14
        j = m.blocks[2][(k-1)*q+g]
        m.lhs[j, j] += 1
    end
    c, (s,) = solutions(m, solve_mme(m))
    check("diag(R₁⁻¹), diag(R₂⁻¹)", rinv,
          [0.2809, 0.7264, 0.0969, 0.1211, 0.6061, 0.4377, 0.2020, 0.3704]; atol = 1e-4)
    check("X'R⁻¹X", Matrix(m.lhs[1:2, 1:2]), [1.2252 0; 0 1.6162]; atol = 1e-4)
    labels = [string.(1:9); "G" .* string.(1:5)]
    show_table(effect = labels, A1 = s[:, 1], B1 = s[:, 1] .+ c[1],
               A2 = s[:, 2], B2 = s[:, 2] .+ c[2])
    check("country effects", c, [7.268, 9.036])
    check("bulls and groups (A)", s,
          [2.604 2.661; 2.176 0.403; 8.059 5.001; -9.865 -5.605; 13.634 9.728;
           -18.086 -13.203; 4.310 3.071; 7.015 4.489; -6.299 -5.059; 0.174 -0.092;
           -0.124 0.126; -0.071 0.264; 0.087 -0.288; -0.067 -0.010])
    check("group solutions sum to zero", sum(s[10:14, :]; dims = 1), [0 0]; atol = 1e-8)

    # partition of the proof of bull 3 (no progeny, DRP in country 1 only)
    Gi = inv(G0)
    αpar = 8 / 11                              # sire and MGS known
    PA3 = 0.5s[7, :] + 0.25(s[2, :] + s[14, :])
    CD3 = [drp[1][3] - c[1], 0]
    ZRZ3 = Diagonal([rinv[3], 0])
    D = ZRZ3 + 2αpar * Gi
    W1, W2 = D \ (2αpar * Gi), D \ ZRZ3
    check("bull 3: PA", PA3, [2.68225, 1.63375])
    # the book prints W₁ = [0.4229 −0.3614; −0.3614 1]; as W₁ + W₂ = I and
    # W₂ = [0.5771 0; 0.3614 0], its (1,2) element is 0
    check("bull 3: W₁", W1, [0.4229 0; -0.3614 1.0]; atol = 2e-4)
    check("bull 3: W₁PA + W₂CD", W1 * PA3 + W2 * CD3, [8.058, 5.000]; atol = 2e-3)
    # Eqn 6.21 is printed with a minus sign; as g²¹/g²² = −g₁₂/g₁₁ it is a
    # plus, which also reproduces the book's own value of 5.001
    check("bull 3 in country 2 by Eqn 6.21",
          PA3[2] + G0[1, 2] / G0[1, 1] * (s[3, 1] - PA3[1]), s[3, 2]; atol = 1e-8)

    # bull 2: DRP in both countries and a maternal grandson (bull 3)
    PA2 = 0.5s[8, :] + 0.25(s[9, :] + s[14, :])
    CD2 = [drp[1][2] - c[1], drp[2][2] - c[2]]
    # the book uses PC = 4â₃ − 2â₇; the exact progeny contribution through a
    # maternal grandson also removes the grandson's MGD group (G5)
    PC2 = 4s[3, :] - 2s[7, :] - s[14, :]
    αprog = 4 / 11                             # mate (sire of the grandson) known
    ZRZ2 = Diagonal([rinv[2], rinv[6]])
    D = ZRZ2 + (2αpar + 0.25αprog) * Gi
    W1, W2, W3 = D \ (2αpar * Gi), D \ ZRZ2, D \ (0.25αprog * Gi)
    check("bull 2: W₂", W2, [0.7867 0.2101; 0.3487 0.3855]; atol = 2e-4)
    check("bull 2: W₁PA + W₂CD + W₃PC = â₂", W1 * PA2 + W2 * CD2 + W3 * PC2, s[2, :];
          atol = 1e-8)
    (; c, s)
end
