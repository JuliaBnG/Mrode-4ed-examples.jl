"""
    ex_06_4() -> NamedTuple

Example 6.4: Multivariate model with zero residual covariance (sex-limited traits in
relatives) (Mrode & Pocrnic, 2023, Section 6.5.1).

# Model Formulation
When traits are sex-limited (e.g. yearling weight recorded only on males and fat yield
recorded only on females), no animal can have records on both traits simultaneously.
The residual covariance between the traits is therefore not estimable and is set
to zero:
```math
R_0 = \\begin{bmatrix} \\sigma_{e1}^2 & 0 \\\\ 0 & \\sigma_{e2}^2 \\end{bmatrix} =
\\begin{bmatrix} 77.0 & 0 \\\\ 0 & 70.0 \\end{bmatrix}
```
while the genetic covariance is non-zero:
```math
G_0 = \\begin{bmatrix} 43.0 & 18.0 \\\\ 18.0 & 30.0 \\end{bmatrix}
```

# Key Concepts Illustrated
1. **One table for the data**: pedigree, sex, HYS and both traits of all 17
   animals in one `DataFrame`, with `missing` where a trait is not recorded;
   the EBV of both traits are added as columns.
2. **Sex-Limited Traits and Correlated Relatives**: Information flows across traits
   strictly through genetic relationships in ``A^{-1}`` and genetic covariance ``G_0``.
   With ``r^{11} = 1/\\sigma_{e1}^2`` and ``r^{22} = 1/\\sigma_{e2}^2``, the MME are
   ```math
   \\begin{bmatrix}
   X_1' r^{11} X_1 & 0 & X_1' r^{11} Z_1 & 0 \\\\
   0 & X_2' r^{22} X_2 & 0 & X_2' r^{22} Z_2 \\\\
   Z_1' r^{11} X_1 & 0 & Z_1' r^{11} Z_1 + A^{-1} g^{11} & A^{-1} g^{12} \\\\
   0 & Z_2' r^{22} X_2 & A^{-1} g^{21} & Z_2' r^{22} Z_2 + A^{-1} g^{22}
   \\end{bmatrix},
   ```
   where ``g^{ij}`` are the elements of ``G_0^{-1}``: the only blocks linking
   the traits are ``A^{-1} g^{12}``. Written out explicitly and compared with
   `mt_stack`, `mt_blocks`, and `mt_rinv`.
3. **Unequal Fixed Effect Levels**: Yearling weight has only 2 HYS levels, while fat yield
   has 3 HYS levels.

# Note on Text Discrepancy
The book prints animal 6 as ``(-5.012, -2.098)``, identical to animal 5.
However, dam 6 has a son (animal 11) with a high yearling weight record (300 kg in HYS 2),
giving her positive evaluations ``(5.012, 2.098)``, as verified by the MME solutions.

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `ebv_yw` and `ebv_fat` added.
- `b`: Fixed HYS effects (levels 1-2 for yearling weight, 1-3 for fat yield).
- `a`: Matrix of estimated breeding values for all 17 animals across both traits.
"""
function ex_06_4()
    println("Example 6.4: different traits recorded on relatives")

    # 1. Pedigree and sex-limited records, one row per animal (IDs are row
    #    numbers): yearling weight on males 9–12, fat yield on females 13–17.
    #    Sex is given in the book for animal 4 and calves 9–17 only.
    m8 = fill(missing, 8)
    ped = DataFrame(
        id = 1:17,
        sex = [missing, missing, missing, "F", missing, missing, missing, missing,
               "M", "M", "M", "M", "F", "F", "F", "F", "F"],
        sire = [0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 3, 1, 3, 2, 2, 3],
        dam = [0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 7, 8, 0, 13, 15],
        hys = [m8; [1, 2, 2, 1, 1, 2, 3, 2, 3]],
        yw = [m8; [375.0, 250.0, 300.0, 450.0]; fill(missing, 5)],
        fat = [m8; fill(missing, 4); [200.0, 160.0, 150.0, 250.0, 175.0]],
    )
    @assert ped.id == 1:nrow(ped)
    println(ped)

    # 2. Covariance matrices (zero environmental covariance)
    G0 = [43.0 18.0; 18.0 30.0]
    R0 = [77.0 0.0; 0.0 70.0]

    # 3. Records and design matrices of each trait: yearling weight has HYS 1–2,
    #    fat yield HYS 1–3
    rw = dropmissing(ped, :yw)                  # males with yearling weight
    rf = dropmissing(ped, :fat)                 # females with fat yield
    y1, y2 = rw.yw, rf.fat
    X1, lv1 = incidence(rw.hys)
    X2, lv2 = incidence(rf.hys)
    Z1 = incidence(rw.id, nrow(ped))
    Z2 = incidence(rf.id, nrow(ped))
    Ai = ainv(ped)

    # 4. MME written out as in the book: no animal has both traits and R₀ is
    #    diagonal (r¹¹ = 1/77, r²² = 1/70), so the traits are linked only
    #    through G₀⁻¹ ⊗ A⁻¹
    r11, r22 = 1 / R0[1, 1], 1 / R0[2, 2]
    Gi = inv(G0)
    O(A, B) = spzeros(size(A, 2), size(B, 2))
    lhs = [
        X1'X1*r11  O(X1, X2)  X1'Z1*r11               O(X1, Z2)
        O(X2, X1)  X2'X2*r22  O(X2, Z1)               X2'Z2*r22
        Z1'X1*r11  O(Z1, X2)  Z1'Z1*r11 + Gi[1, 1]*Ai Gi[1, 2]*Ai
        O(Z2, X1)  Z2'X2*r22  Gi[2, 1]*Ai             Z2'Z2*r22 + Gi[2, 2]*Ai
    ]
    rhs = [X1'y1 * r11; X2'y2 * r22; Z1'y1 * r11; Z2'y2 * r22]
    check("G₀⁻¹", Gi, [0.0311 -0.0186; -0.0186 0.0445]; atol = 1e-4)

    # 4b. The same equations with mt_stack, mt_blocks and mt_rinv
    rec = dropmissing(ped, :hys)                # calves with a record
    Y = Matrix(rec[:, [:yw, :fat]])
    y, obs = mt_stack(Y)
    Xw, _ = incidence(rec.hys; levels = lv1)    # no HYS 3 for yearling weight
    Xf, _ = incidence(rec.hys; levels = lv2)
    Z = incidence(rec.id, nrow(ped))
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [Xw, Xf])),
        [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0; name = "animal")];
        Rinv = mt_rinv(R0, obs),
    )
    check("mme() LHS = explicit form", m.lhs, lhs; atol = 1e-12)
    check("mme() RHS = explicit form", m.rhs, rhs; atol = 1e-12)

    # 5. Solve MME
    x = lhs \ rhs
    b = x[m.blocks[1]]
    a = reshape(x[m.blocks[2]], :, 2)           # rows: animals, cols: weight and fat
    ped.ebv_yw, ped.ebv_fat = a[:, 1], a[:, 2]

    println("\nHYS effects:")
    show_table(trait = [fill("weight", 2); fill("fat", 3)], hys = [lv1; lv2], solution = b)
    check("HYS, yearling weight", b[1:2], [412.26, 276.21]; atol = 6e-3)
    check("HYS, fat", b[3:5], [194.03, 204.77, 161.66]; atol = 5e-3)

    # Note on book typo:
    # The book prints animal 6 as (-5.012, -2.098), a copy of animal 5; dam 6
    # has a son (11) with a high record, so her values are positive (+5.012, +2.098)
    check(
        "animals",
        a,
        [
            -3.365 1.258;
            -1.489 3.774;
            4.237 -1.687;
            -6.940 -1.572;
            -5.012 -2.098;
            5.012 2.098;
            2.137 3.561;
            -4.274 -7.123;
            -12.162 -3.091;
            -8.263 -1.260;
            5.836 3.776;
            12.632 3.558;
            1.523 5.971;
            -4.292 -11.527;
            -1.870 0.011;
            4.290 11.995;
            2.684 1.663
        ],
    )

    println("\nData with solutions:")
    show_table(ped)

    (; ped, b, a)
end
