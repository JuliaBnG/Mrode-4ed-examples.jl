"""
    ex_06_5() -> NamedTuple

Example 6.5: Multi-trait Across-Country Evaluation (MACE) of bulls
(Mrode & Pocrnic, 2023, Section 6.5.2; Schaeffer, 1994).

# Model Formulation
MACE evaluates bulls internationally using national genetic evaluation summaries
(Deregressed Proofs, DRP) rather than raw individual phenotypes:
```math
y_i = X_i c_i + Z_i s_i + e_i
```
where:
- ``y_i`` is the vector of DRP in country ``i`` (countries 1 and 2).
- ``c_i`` is the fixed country mean.
- ``s_i`` is the vector of sire transmitting abilities across countries, modeled with
  sire–maternal-grandsire (sire–MGS) relationships and phantom parent groups.
- ``\\mathrm{var}(s) = G_0 \\otimes A_{\\mathrm{smgs}}``.
- ``\\mathrm{var}(e_i) = \\mathrm{diag}(\\sigma_{ei}^2 / \\mathrm{EDC}_{ik})``,
  weighted inversely by each bull's Effective Daughter Contribution (EDC).

# Key Concepts Illustrated
1. **Tables**: the sire–MGS pedigree of the bulls (one row per bull), and the
   national proofs (one row per bull and country, with DRP and EDC). Proofs
   are added as columns of the bull table, group solutions to a table of
   groups.
2. **Sire–MGS Model with Phantom Parent Groups**:
   Constructed via `ainv_smgs` with group equations stabilized.
3. **Heterogeneous residuals**: ``R^{-1} = \\mathrm{diag}(\\mathrm{EDC}_{ik}/\\sigma_{ei}^2)``
   differs for every proof, so the MME are in the general form
   ```math
   \\begin{bmatrix}
   X'R^{-1}X & X'R^{-1}Z \\\\
   Z'R^{-1}X & Z'R^{-1}Z + G_0^{-1} \\otimes A_{\\mathrm{smgs}}^{-1} + S
   \\end{bmatrix},
   ```
   with ``Z = \\mathrm{blockdiag}(Z_1, Z_2)`` and ``S`` adding 1 to the
   diagonals of the group equations. Written out explicitly and compared
   with `mme(...; Rinv)`.
4. **Decomposition of MACE Proofs** (Eqns 6.19–6.21):
   Decomposes proofs into Parent Average (PA), Country Daughter Yield Deviation (CD),
   and Progeny Contribution (PC) through daughters and maternal grandsons.

# Notes on Text Discrepancies
1. The book prints ``W_1`` for bull 3 with a spurious (1, 2) entry of ``-0.3614``.
   Since ``W_1 + W_2 = I`` and ``W_2 = [0.5771, 0; 0.3614, 0]``, the (1, 2) element is 0.
2.  Eqn 6.21 in the book is printed with a minus sign; because
   ``g^{21} / g^{22} = -g_{12} / g_{11}``, the formula requires a plus sign, which also
   reproduces the book's own value of 5.001.
3.  The book computes PC for bull 2 as ``4\\hat{a}_3 - 2\\hat{a}_7``; the exact progeny
   contribution through a maternal grandson also removes the grandson's maternal granddam
   group (G5).

# Returns
A `NamedTuple` with fields:
- `ped`: The bulls with their proofs `A1`, `A2` and `B1`, `B2` (A plus the
  country mean) added.
- `groups`: The phantom groups with their solutions `A1`, `A2`.
- `rec`: The national proofs with the residual precisions `rinv` added.
- `c`: Fixed country effect solutions.
- `s`: Matrix of bull and group solutions for countries 1 and 2.
"""
function ex_06_5()
    println("Example 6.5: MACE for two countries")

    # 1. Sire–MGS pedigree of bulls 1–9 (IDs are row numbers); unknown
    #    ancestors are assigned to phantom groups G1–G5 (coded -1 … -5)
    ped = DataFrame(
        bull = 1:9,
        sire = [7, 8, 7, 1, 8, 1, -1, -1, -1],
        mgs = [-3, 9, 2, -2, -3, 9, -2, -2, -3],
        mgd = [-5, -5, -5, -5, -4, -4, -4, -4, -4],
    )
    @assert ped.bull == 1:nrow(ped)
    groups = DataFrame(group = "G" .* string.(1:5))

    # 2. National evaluation summaries, one row per proof: deregressed proof
    #    (DRP) and effective daughter contribution (EDC) of a bull in a country
    rec = DataFrame(
        country = [1, 1, 1, 1, 2, 2, 2, 2],
        bull = [1, 2, 3, 4, 1, 2, 5, 6],
        edc = [58, 150, 20, 25, 90, 65, 30, 55],
        drp = [9.7229, 9.9717, 19.2651, -8.5711, 14.5088, 7.7594, 23.9672, -9.6226],
    )
    @assert issorted(rec.country)                # countries stacked as traits
    σ²e = [206.5, 148.5]
    G0 = [20.5 12.839; 12.839 9.5]
    println(rec)

    # 3. Residual precision: each proof is weighted by its EDC, R⁻¹ = diag(EDC/σ²e)
    rec.rinv = rec.edc ./ σ²e[rec.country]
    check(
        "diag(R₁⁻¹), diag(R₂⁻¹)",
        rec.rinv,
        [0.2809, 0.7264, 0.0969, 0.1211, 0.6061, 0.4377, 0.2020, 0.3704];
        atol = 1e-4,
    )

    # 4. Design matrices: country means, and bulls + groups in each country
    Ai = ainv_smgs(ped)                         # 9 bulls + 5 groups
    q = size(Ai, 1)
    y = rec.drp
    X, _ = incidence(rec.country)
    r1, r2 = (groupby(rec, :country)...,)
    Zb = blockdiag(incidence(r1.bull, q), incidence(r2.bull, q))

    # 5. MME written out. R is diagonal but heterogeneous, so only the general
    #    R⁻¹ form applies. Phantom groups are not estimable from the data: 1 is
    #    added to the diagonals of their equations (groups random with variance 1).
    Ri = Diagonal(rec.rinv)
    isgroup = [falses(nrow(ped)); trues(nrow(groups))]
    stab = kron(I(2), Diagonal(Float64.(isgroup)))
    #! format: off
    lhs = [
        X'Ri*X   X'Ri*Zb
        Zb'Ri*X  Zb'Ri*Zb + kron(inv(G0), Ai) + stab
    ]
    #! format: on
    rhs = [X'Ri*y; Zb'Ri*y]
    check("X'R⁻¹X", Matrix(lhs[1:2, 1:2]), [1.2252 0; 0 1.6162]; atol = 1e-4)

    Z1 = [incidence(r1.bull, q); spzeros(nrow(r2), q)]
    Z2 = [spzeros(nrow(r1), q); incidence(r2.bull, q)]
    m = mme(y, X, [RandomEffect([Z1, Z2], Ai, G0; name = "bull+group")]; Rinv = rec.rinv)
    check(
        "mme() LHS + group stabilization = explicit form",
        m.lhs + blockdiag(spzeros(2, 2), sparse(stab)),
        lhs;
        atol = 1e-12,
    )

    # 6. Solve MME
    x = lhs \ rhs
    c = x[m.blocks[1]]
    s = reshape(x[m.blocks[2]], :, 2)           # rows: bulls 1–9, G1–G5; cols: countries

    # Proofs on each country's scale: A = bull solution, B = A + country mean
    ped.A1, ped.A2 = s[1:9, 1], s[1:9, 2]
    ped.B1, ped.B2 = ped.A1 .+ c[1], ped.A2 .+ c[2]
    groups.A1, groups.A2 = s[10:14, 1], s[10:14, 2]
    println("\nCountry effects: ", round.(c; digits = 3))
    println("\nBulls:")
    show_table(ped)
    println("\nGroups:")
    show_table(groups)
    check("country effects", c, [7.268, 9.036])
    check(
        "bulls and groups (A)",
        s,
        [
            2.604 2.661;
            2.176 0.403;
            8.059 5.001;
            -9.865 -5.605;
            13.634 9.728;
            -18.086 -13.203;
            4.310 3.071;
            7.015 4.489;
            -6.299 -5.059;
            0.174 -0.092;
            -0.124 0.126;
            -0.071 0.264;
            0.087 -0.288;
            -0.067 -0.010
        ],
    )
    check("group solutions sum to zero", sum(s[10:14, :]; dims = 1), [0 0]; atol = 1e-8)

    # 6. Partition of the proof of bull 3 (no progeny, DRP in country 1 only)
    Gi = inv(G0)
    αpar = 8 / 11                              # sire and MGS known
    PA3 = 0.5s[7, :] + 0.25(s[2, :] + s[14, :])
    CD3 = [rec.drp[3] - c[1], 0.0]
    ZRZ3 = Diagonal([rec.rinv[3], 0.0])
    D = ZRZ3 + 2αpar * Gi
    W1, W2 = D \ (2αpar * Gi), D \ ZRZ3
    check("bull 3: PA", PA3, [2.68225, 1.63375])
    # The book prints W₁ = [0.4229 -0.3614; -0.3614 1]; as W₁ + W₂ = I and
    # W₂ = [0.5771 0; 0.3614 0], its (1,2) element is 0
    check("bull 3: W₁", W1, [0.4229 0; -0.3614 1.0]; atol = 2e-4)
    check("bull 3: W₁PA + W₂CD", W1 * PA3 + W2 * CD3, [8.058, 5.000]; atol = 2e-3)
    # Eqn 6.21 is printed with a minus sign; as g²¹/g²² = -g₁₂/g₁₁ it is a
    # plus, which also reproduces the book's own value of 5.001
    check(
        "bull 3 in country 2 by Eqn 6.21",
        PA3[2] + G0[1, 2] / G0[1, 1] * (s[3, 1] - PA3[1]),
        s[3, 2];
        atol = 1e-8,
    )

    # 7. Partition for bull 2: DRP in both countries and a maternal grandson (bull 3)
    PA2 = 0.5s[8, :] + 0.25(s[9, :] + s[14, :])
    CD2 = [rec.drp[2] - c[1], rec.drp[6] - c[2]]
    # The book uses PC = 4â₃ - 2â₇; the exact progeny contribution through a
    # maternal grandson also removes the grandson's MGD group (G5)
    PC2 = 4s[3, :] - 2s[7, :] - s[14, :]
    αprog = 4 / 11                             # mate (sire of grandson) known
    ZRZ2 = Diagonal([rec.rinv[2], rec.rinv[6]])
    D = ZRZ2 + (2αpar + 0.25αprog) * Gi
    W1, W2, W3 = D \ (2αpar * Gi), D \ ZRZ2, D \ (0.25αprog * Gi)
    check("bull 2: W₂", W2, [0.7867 0.2101; 0.3487 0.3855]; atol = 2e-4)
    check(
        "bull 2: W₁PA + W₂CD + W₃PC = â₂",
        W1 * PA2 + W2 * CD2 + W3 * PC2,
        s[2, :];
        atol = 1e-8,
    )

    (; ped, groups, rec, c, s)
end
