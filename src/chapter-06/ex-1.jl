"""
    univariate(y, X, Z, Ai, σ²a, σ²e) -> (x, m)

Helper function to fit a univariate animal model for one trait of the beef calves,
used to compare univariate and multi-trait BLUP evaluations.
"""
function univariate(y, X, Z, Ai, σ²a, σ²e)
    m = mme(y, X, [RandomEffect(Z, Ai, σ²a)]; σ²e)
    x = solve_mme(m)
    x, m
end

"""
    ex_06_1() -> NamedTuple

Example 6.1: Multivariate animal model with equal design matrices and no missing records
(Mrode & Pocrnic, 2023, Sections 6.1–6.2).

# Model Formulation
For two traits measured on the same individuals (weaning gain WWG and post-weaning gain
PWG):
```math
\\begin{bmatrix} y_1 \\\\ y_2 \\end{bmatrix} =
\\begin{bmatrix} X_1 & 0 \\\\ 0 & X_2 \\end{bmatrix} \\begin{bmatrix} b_1 \\\\ b_2
    \\end{bmatrix} +
\\begin{bmatrix} Z_1 & 0 \\\\ 0 & Z_2 \\end{bmatrix} \\begin{bmatrix} a_1 \\\\ a_2
    \\end{bmatrix} +
\\begin{bmatrix} e_1 \\\\ e_2 \\end{bmatrix}
```
where:
- Design matrices are identical: ``X_1 = X_2 = X`` and ``Z_1 = Z_2 = Z``.
- Genetic (co)variance: ``\\mathrm{var}(a) = G_0 \\otimes A`` with
  ``G_0 = \\begin{bmatrix} 20.0 & 18.0 \\\\ 18.0 & 40.0 \\end{bmatrix}``
  (genetic correlation ``r_g = 18 / \\sqrt{20 \\times 40} = 0.636``).
- Residual (co)variance: ``\\mathrm{var}(e) = R_0 \\otimes I`` with
  ``R_0 = \\begin{bmatrix} 40.0 & 11.0 \\\\ 11.0 & 30.0 \\end{bmatrix}``
  (residual correlation ``r_e = 11 / \\sqrt{40 \\times 30} = 0.318``).

# Key Concepts Illustrated
1. **One table for the data**: pedigree, sex and both traits of all 8 animals
   in one `DataFrame`; all animal-level results (multi-trait and univariate
   EBV, PEV, reliabilities) are added as columns.
2. **Multi-Trait BLUP**: with equal design matrices and no missing records,
   and traits stacked (``y = \\mathrm{vec}(Y)``), the MME are
   ```math
   \\begin{bmatrix}
   R_0^{-1} \\otimes X'X & R_0^{-1} \\otimes X'Z \\\\
   R_0^{-1} \\otimes Z'X & R_0^{-1} \\otimes Z'Z + G_0^{-1} \\otimes A^{-1}
   \\end{bmatrix}
   \\begin{bmatrix}\\hat{b} \\\\ \\hat{a}\\end{bmatrix} =
   \\begin{bmatrix}(R_0^{-1} \\otimes X') y \\\\ (R_0^{-1} \\otimes Z') y\\end{bmatrix},
   ```
   written out with `kron` and compared with the general set-up by
   `mt_stack`, `mt_blocks`, and `mt_rinv`, which also handles missing records
   and different design matrices per trait.
3. **Comparison with Univariate BLUP**: Demonstrating changes in EBV ranking and
   information borrowing due to genetic and environmental covariances.
4. **Multi-Trait Reliabilities** (Section 6.2.4):
   Prediction error variance matrix ``P = \\mathrm{diag}(C^{22})``, with
   ``r_{ij}^2 = 1 - P_{ij} / G_{0, jj}``.
5. **Multi-Trait EBV Partition** (Section 6.2.3):
   Matrix decomposition ``\\hat{a}_i = W_1 \\mathrm{PA}_i + W_2 \\mathrm{YD}_i``.

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `ebv_wwg`, `ebv_pwg`, `uni_wwg`, `uni_pwg`,
  `pev_wwg`, `pev_pwg`, `r2_wwg` and `r2_pwg` added.
- `b`: Fixed sex effects for WWG and PWG.
- `a`: Matrix of estimated breeding values (rows: 8 animals, cols: WWG and PWG).
- `P`: Prediction error variances (PEV) for each animal and trait.
"""
function ex_06_1()
    println("Example 6.1: bivariate analysis of WWG and PWG")

    # 1. Pedigree and records of both traits, one row per animal
    #    (IDs are row numbers; founders 1–3 have no records)
    no3 = fill(missing, 3)
    ped = DataFrame(
        calf = 1:8,
        sex = [no3; ["M", "F", "F", "M", "M"]],
        sire = [0, 0, 0, 1, 3, 1, 4, 3],
        dam = [0, 0, 0, 0, 2, 2, 5, 6],
        wwg = [no3; [4.5, 2.9, 3.9, 3.5, 5.0]],
        pwg = [no3; [6.8, 5.0, 6.8, 6.0, 7.5]],
    )
    @assert ped.calf == 1:nrow(ped)
    rec = dropmissing(ped, [:wwg, :pwg])        # no missing records in this example
    calf = rec.calf
    Y = Matrix(rec[:, [:wwg, :pwg]])            # col 1: WWG, col 2: PWG
    println(ped)

    # 2. Covariance matrices
    G0 = [20.0 18.0; 18.0 40.0]                         # genetic covariance
    R0 = [40.0 11.0; 11.0 30.0]                         # residual covariance

    # 3. Incidence matrices, the same for both traits
    Ai = ainv(ped)
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Z = incidence(calf, nrow(ped))

    # 3a. MME written out. With equal design matrices and no missing records,
    #     traits stacked (y = vec(Y)), var(e) = R₀ ⊗ I and var(a) = G₀ ⊗ A:
    #     [R₀⁻¹⊗X'X  R₀⁻¹⊗X'Z; R₀⁻¹⊗Z'X  R₀⁻¹⊗Z'Z + G₀⁻¹⊗A⁻¹]
    Ri = inv(R0)
    #! format: off
    lhs = [
        kron(Ri, X'X)  kron(Ri, X'Z)
        kron(Ri, Z'X)  kron(Ri, Z'Z) + kron(inv(G0), Ai)
    ]
    #! format: on
    rhs = [kron(Ri, X'); kron(Ri, Z')] * vec(Y)

    # 3b. The same equations with mt_stack, mt_blocks and mt_rinv, which also
    #     handle missing records and different design matrices per trait
    y, obs = mt_stack(Y)
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [X, X])),
        [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0; name = "animal")];
        Rinv = mt_rinv(R0, obs),
    )
    check("mme() LHS = Kronecker form", m.lhs, lhs; atol = 1e-12)
    check("mme() RHS = Kronecker form", m.rhs, rhs; atol = 1e-12)

    # 4. Solve multi-trait MME and compare with univariate models
    x = lhs \ rhs
    b = x[m.blocks[1]]
    a = reshape(x[m.blocks[2]], :, 2)           # rows: animals, cols: WWG and PWG
    xw, _ = univariate(Y[:, 1], X, Z, Ai, 20.0, 40.0)
    xp, _ = univariate(Y[:, 2], X, Z, Ai, 40.0, 30.0)

    println("\nSex effects:")
    show_table(
        sex = ["M", "F"],
        WWG = b[1:2],
        PWG = b[3:4],
        WWG_uni = xw[1:2],
        PWG_uni = xp[1:2],
    )
    ped.ebv_wwg, ped.ebv_pwg = a[:, 1], a[:, 2]
    ped.uni_wwg, ped.uni_pwg = xw[3:end], xp[3:end]
    check("sex (WWG, PWG)", b, [4.361, 3.397, 6.800, 5.880])
    check(
        "animals, WWG",
        a[:, 1],
        [0.151, -0.015, -0.078, -0.010, -0.270, 0.276, -0.316, 0.244],
    )
    check(
        "animals, PWG",
        a[:, 2],
        [0.280, -0.008, -0.170, -0.013, -0.478, 0.517, -0.479, 0.392],
    )
    check(
        "univariate WWG",
        xw[3:end],
        [0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.249, 0.183],
    )
    check(
        "univariate PWG",
        [xp[1:2]; xp[3:end]],
        [6.798, 5.879, 0.277, -0.005, -0.171, -0.013, -0.471, 0.514, -0.464, 0.384],
    )

    # 5. Reliabilities r² = (g_jj - PEV_ij) / g_jj (Section 6.2.4)
    P = pev(m, 1)
    r² = 1 .- P ./ diag(G0)'

    ped.pev_wwg, ped.pev_pwg = P[:, 1], P[:, 2]
    ped.r2_wwg, ped.r2_pwg = r²[:, 1], r²[:, 2]
    check(
        "PEV (diagonals of C⁻¹)",
        P,
        [
            18.606 35.904;
            19.596 38.768;
            17.893 33.799;
            16.506 29.727;
            16.541 29.865;
            17.152 31.504;
            17.115 31.364;
            16.285 29.160
        ];
        atol = 3e-3,
    )
    check(
        "reliabilities",
        r²,
        [
            0.070 0.102;
            0.020 0.031;
            0.105 0.155;
            0.175 0.257;
            0.173 0.253;
            0.142 0.212;
            0.144 0.216;
            0.186 0.271
        ],
    )

    # 6. Multi-trait partition: â₈ = W₁ PA + W₂ YD (Section 6.2.3)
    Ri = inv(R0)
    ZRZ = [i in calf ? Ri : zeros(2, 2) for i = 1:nrow(ped)]
    yd = zeros(nrow(ped), 2)
    yd[calf, :] = Y - [X * b[1:2] X * b[3:4]]
    p = ebv_partition_mt(ped.sire, ped.dam, G0, ZRZ, yd, a)

    check("calf 8: W₁", p.W1[8], [0.8476 -0.1191; -0.0237 0.6092]; atol = 5e-4)
    check("calf 8: PA and YD", [p.PA[8] yd[8, :]], [0.099 0.639; 0.1735 0.700])
    check(
        "calf 8: W₁PA + W₂YD = â₈",
        p.W1[8] * p.PA[8] + p.W2[8] * yd[8, :],
        a[8, :];
        atol = 1e-10,
    )

    println("\nData with multi-trait and univariate EBV:")
    show_table(
        ped[:, [:calf, :sire, :dam, :wwg, :pwg, :ebv_wwg, :uni_wwg, :ebv_pwg, :uni_pwg]],
    )
    println("\nPrediction error variances and reliabilities:")
    show_table(ped[:, [:calf, :pev_wwg, :pev_pwg, :r2_wwg, :r2_pwg]])

    (; ped, b, a, P)
end
