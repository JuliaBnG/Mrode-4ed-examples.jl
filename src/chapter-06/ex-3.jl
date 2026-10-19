"""
    ex_06_3() -> NamedTuple

Example 6.3: Multivariate animal model with unequal design matrices and multi-trait DYD
(Mrode & Pocrnic, 2023, Section 6.4).

# Model Formulation
Lactation fat yields in parities 1 and 2 are analyzed as two distinct, correlated traits.
Unlike Example 6.1, the fixed effects design matrices differ across traits:
```math
\\begin{bmatrix} y_1 \\\\ y_2 \\end{bmatrix} =
\\begin{bmatrix} X_1 & 0 \\\\ 0 & X_2 \\end{bmatrix} \\begin{bmatrix} b_1 \\\\ b_2
    \\end{bmatrix} +
\\begin{bmatrix} Z_1 & 0 \\\\ 0 & Z_2 \\end{bmatrix} \\begin{bmatrix} a_1 \\\\ a_2
    \\end{bmatrix} +
\\begin{bmatrix} e_1 \\\\ e_2 \\end{bmatrix}
```
where:
- ``X_1`` models HYS effects in parity 1, and ``X_2`` models HYS effects in parity 2.
- ``G_0 = \\begin{bmatrix} 35.0 & 28.0 \\\\ 28.0 & 30.0 \\end{bmatrix}``,
  ``R_0 = \\begin{bmatrix} 65.0 & 27.0 \\\\ 27.0 & 70.0 \\end{bmatrix}``.

# Key Concepts Illustrated
1. **From repeated records to traits**: the lactation records of Example 5.1
   (one row per record) are reshaped with `unstack` to one row per cow, with
   fat yield and HYS of each parity as columns. Animal-level results are
   added as columns of the pedigree.
2. **Unequal Fixed Effects Design Matrices**: HYS 1–2 occur only in parity 1
   and HYS 3–4 only in parity 2, so each trait has its own ``X_t``. The MME
   of Example 6.1 hold with ``\\mathrm{blockdiag}(X_1, X_2)`` in place of
   ``I_2 \\otimes X``:
   ```math
   \\begin{bmatrix}
   X_b' R^{-1} X_b & X_b' R^{-1} Z_b \\\\
   Z_b' R^{-1} X_b & Z_b' R^{-1} Z_b + G_0^{-1} \\otimes A^{-1}
   \\end{bmatrix},
   \\quad X_b = \\begin{bmatrix} X_1 & 0 \\\\ 0 & X_2 \\end{bmatrix},
   \\quad Z_b = I_2 \\otimes Z,
   \\quad R^{-1} = R_0^{-1} \\otimes I,
   ```
   written out explicitly and compared with `mt_stack`, `mt_blocks`, and
   `mt_rinv`.
3. **Multi-Trait Daughter Yield Deviation (DYD)** (Section 6.4.2):
   Computes matrix weights ``W_2^{\\mathrm{prog}}`` and ``M_3`` for calculating sire DYD
   from correlated multi-trait records of daughters.

# Note on Text Discrepancy
The book prints ``\\mathrm{YD}_{4,1} = 25.7`` although ``201 - 175.7 = 25.3``.
Correcting this arithmetic slip yields ``\\mathrm{DYD}_1 = [24.254, 32.154]'``
instead of the book's printed ``[24.521, 32.154]'``.

# Returns
A `NamedTuple` with fields:
- `ped`: The pedigree with columns `ebv_fat1`, `ebv_fat2`, `uni_fat1`, `uni_fat2`,
  `dyd_fat1` and `dyd_fat2` added.
- `cows`: The records reshaped to one row per cow.
- `b`: Fixed HYS effect solutions for parities 1 and 2.
- `a`: Matrix of estimated breeding values (rows: 8 animals, cols: parities 1 and 2).
- `partition`: Results of the multi-trait partition and sire DYD calculation.
"""
function ex_06_3()
    println("Example 6.3: fat yield in parities 1 and 2 as different traits")

    # 1. Pedigree, one row per animal (IDs are row numbers), and the lactation
    #    records of Example 5.1, one row per record
    ped = DataFrame(id = 1:8, sire = [0, 0, 0, 1, 3, 1, 3, 1], dam = [0, 0, 0, 2, 2, 5, 4, 7])
    @assert ped.id == 1:nrow(ped)
    rec = DataFrame(
        cow = repeat(4:8; inner = 2),
        parity = repeat([1, 2], 5),
        hys = [1, 3, 1, 4, 2, 3, 1, 3, 2, 4],
        fy = [201.0, 280.0, 150.0, 200.0, 160.0, 190.0, 180.0, 250.0, 285.0, 300.0],
    )

    # 2. Parities as different traits: reshape to one row per cow. HYS 1–2 occur
    #    only in parity 1 and HYS 3–4 only in parity 2, so each trait has its
    #    own fixed effects.
    cows = innerjoin(
        unstack(rec, :cow, :parity, :fy; renamecols = p -> "fat$p"),
        unstack(rec, :cow, :parity, :hys; renamecols = p -> "hys$p");
        on = :cow,
    )
    println(cows)
    cow = cows.cow
    Y = Matrix{Float64}(cows[:, [:fat1, :fat2]])

    # 3. Covariance matrices
    R0 = [65.0 27.0; 27.0 70.0]
    G0 = [35.0 28.0; 28.0 30.0]

    # 4. Unequal design matrices: X₁ for HYS 1–2 (parity 1), X₂ for HYS 3–4 (parity 2)
    Ai = ainv(ped)
    X1, lv1 = incidence(cows.hys1)
    X2, lv2 = incidence(cows.hys2)
    Z = incidence(cow, nrow(ped))

    # 4a. MME written out: as in Example 6.1, but with blockdiag(X₁, X₂)
    #     instead of I₂ ⊗ X. No missing records, so R⁻¹ = R₀⁻¹ ⊗ I.
    Xb = blockdiag(X1, X2)                      # [X₁ 0; 0 X₂]
    Zb = blockdiag(Z, Z)                        # [Z 0; 0 Z]
    Ri = kron(inv(R0), I(length(cow)))
    lhs = [
        Xb'Ri*Xb  Xb'Ri*Zb
        Zb'Ri*Xb  Zb'Ri*Zb + kron(inv(G0), Ai)
    ]
    rhs = [Xb'Ri*vec(Y); Zb'Ri*vec(Y)]

    # 4b. The same equations with mt_stack, mt_blocks and mt_rinv
    y, obs = mt_stack(Y)
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [X1, X2])),
        [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0; name = "animal")];
        Rinv = mt_rinv(R0, obs),
    )
    check("mme() LHS = blockdiag form", m.lhs, lhs; atol = 1e-12)
    check("mme() RHS = blockdiag form", m.rhs, rhs; atol = 1e-12)

    # 5. Solve MME and fit univariate benchmarks
    x = lhs \ rhs
    b = x[m.blocks[1]]
    a = reshape(x[m.blocks[2]], :, 2)           # rows: animals, cols: parities 1 and 2
    uni = [
        solve_mme(
            mme(Y[:, t], (X1, X2)[t], [RandomEffect(Z, Ai, G0[t, t])]; σ²e = R0[t, t]),
        ) for t = 1:2
    ]

    println("\nHYS effects:")
    show_table(
        hys = [lv1; lv2],
        parity = [1, 1, 2, 2],
        solution = b,
        univariate = [uni[1][1:2]; uni[2][1:2]],
    )
    ped.ebv_fat1, ped.ebv_fat2 = a[:, 1], a[:, 2]
    ped.uni_fat1, ped.uni_fat2 = uni[1][3:end], uni[2][3:end]
    check("HYS (FAT1, FAT2)", b, [175.7, 219.6, 243.2, 240.6]; atol = 0.051)
    check(
        "animals",
        a,
        [
            8.969 8.840;
            -2.999 -2.777;
            -5.970 -6.063;
            11.754 11.658;
            -16.253 -15.824;
            -17.314 -15.719;
            8.690 8.138;
            22.702 20.931
        ],
    )
    check(
        "univariate FAT1 animals",
        uni[1][3:end],
        [6.933, -2.59, -4.341, 9.103, -12.992, -15.197, 7.566, 19.417];
        atol = 5e-3,
    )
    check(
        "univariate FAT2 animals",
        uni[2][3:end],
        [8.665, -2.244, -6.422, 12.197, -15.563, -11.149, 7.696, 15.560],
    )

    # 5. Multi-trait DYD of sire 1 from daughters 4, 6, and 8 (Section 6.4.2)
    R0i = inv(R0)
    ZRZ = [i in cow ? R0i : zeros(2, 2) for i = 1:nrow(ped)]
    yd = zeros(nrow(ped), 2)
    yd[cow, :] = Y - [X1 * b[1:2] X2 * b[3:4]]
    p = ebv_partition_mt(ped.sire, ped.dam, G0, ZRZ, yd, a)
    ped.dyd_fat1 = replace(first.(p.DYD), NaN => missing)    # missing: no progeny
    ped.dyd_fat2 = replace(last.(p.DYD), NaN => missing)
    W2prog = (R0i + 2inv(G0)) \ R0i
    check("W2prog", W2prog, [0.1713 0.0821; 0.1078 0.1244]; atol = 2e-4)

    # Note on book arithmetic correction:
    # The book prints YD₄₁ = 25.7 although 201 - 175.7 = 25.3; with that slip
    # corrected its own numbers give DYD₁ = (24.254, 32.154) instead of
    # (24.521, 32.154), and M₃DYD = (7.388, 7.358) instead of (7.439, 7.387).
    check("DYD of sire 1 (book's slip corrected)", p.DYD[1], [24.254, 32.154]; atol = 0.05)
    check("M₃ of sire 1", p.M3[1], [0.1937 0.0836; 0.1099 0.1459]; atol = 3e-4)
    check(
        "M₃ DYD (no granddaughter information)",
        p.M3[1] * p.DYD[1],
        [7.388, 7.358];
        atol = 0.02,
    )

    println("\nPedigree with multi-trait and univariate EBV, and progeny yield deviations:")
    show_table(ped)

    (; ped, cows, b, a, partition = p)
end
