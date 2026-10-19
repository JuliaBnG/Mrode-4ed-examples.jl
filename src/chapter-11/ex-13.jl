"""
    ex_11_13() -> NamedTuple

Example 11.13: Multivariate GBLUP for beef calves
(Mrode & Pocrnic, 2023, Section 11.8).

# Model Formulation
A multi-trait genomic model extending Example 6.1 to genomic evaluations:
```math
\\begin{bmatrix} y_1 \\\\ y_2 \\end{bmatrix} =
(I_2 \\otimes X) \\begin{bmatrix} b_1 \\\\ b_2 \\end{bmatrix} +
(I_2 \\otimes Z) \\begin{bmatrix} a_{g1} \\\\ a_{g2} \\end{bmatrix} +
\\begin{bmatrix} e_1 \\\\ e_2 \\end{bmatrix}
```
where:
- Two traits (WWG and PWG) are evaluated simultaneously on 5 calves (animals 4 to 8).
- Genotypic data for 10 SNPs are used to construct the genomic relationship matrix ``G``
  via VanRaden (2008) method 1.
- Random genomic effects have covariance ``\\mathrm{var}(a_g) = G_0 \\otimes G`` with
  ``G_0 = \\begin{bmatrix} 20.0 & 18.0 \\\\ 18.0 & 40.0 \\end{bmatrix}``.
- Residual errors have covariance ``\\mathrm{var}(e) = R_0 \\otimes I`` with
  ``R_0 = \\begin{bmatrix} 40.0 & 11.0 \\\\ 11.0 & 30.0 \\end{bmatrix}``.

# Key Concepts Illustrated
1. **Same MME as Example 6.1**, with ``Z = I`` and ``G`` in place of ``A``:
   ``[R_0^{-1} \\otimes X'X\\ R_0^{-1} \\otimes X';\\ R_0^{-1} \\otimes X\\
   R_0^{-1} \\otimes I + G_0^{-1} \\otimes G^{-1}]``, written out with `kron` and
   compared with `mme`.
2. **Tables**: the calves with both traits; their DGV and, for comparison, the
   pedigree-based multi-trait EBV of Example 6.1 are added as columns.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed sex effects for WWG and PWG.
- `a`: Direct genomic values (DGVs) for calves 4 to 8 for both traits.
- `calves`: The calves with DGV and pedigree-based EBV of both traits.
"""
function ex_11_13()
    println("Example 11.13: multivariate genomic model")

    # 1. Genotypes of calves 4–8 (5 animals, 10 SNPs)
    M = [
        0 2 1 1 1 2 0 1 1 1
        2 1 2 1 1 0 1 0 2 0
        1 2 1 0 0 1 1 1 2 0
        1 2 2 1 0 1 1 1 2 1
        1 1 1 1 1 0 1 1 2 0
    ]
    calves = DataFrame(
        calf = 4:8,
        sex = ["M", "F", "F", "M", "M"],
        wwg = [4.5, 2.9, 3.9, 3.5, 5.0],
        pwg = [6.8, 5.0, 6.8, 6.0, 7.5],
    )
    Y = Matrix(calves[:, [:wwg, :pwg]])

    # 2. Covariance matrices
    G0 = [20.0 18.0; 18.0 40.0]
    R0 = [40.0 11.0; 11.0 30.0]

    # 3. Construct genomic relationship matrix G
    _, p, k = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)
    check("2Σpq", k, 4.08; atol = 0.005)
    check(
        "G",
        G,
        [
            1.137 -0.725 -0.088 0.010 -0.333;
            -0.725 0.843 -0.235 -0.137 0.255;
            -0.088 -0.235 0.402 0.010 -0.088;
            0.010 -0.137 0.010 0.353 -0.235;
            -0.333 0.255 -0.088 -0.235 0.402
        ],
    )

    # 4. Multi-trait GBLUP MME, written out as in Example 6.1 with Z = I (each
    #    calf has its own record) and G in place of A:
    #    [R₀⁻¹⊗X'X R₀⁻¹⊗X'; R₀⁻¹⊗X R₀⁻¹⊗I + G₀⁻¹⊗G⁻¹]
    X, _ = incidence(calves.sex; levels = ["M", "F"])
    Gi = inv(G + 0.01I)
    Ri = inv(R0)
    n = nrow(calves)
    #! format: off
    lhs = [
        kron(Ri, X'X)  kron(Ri, X')
        kron(Ri, X)    kron(Ri, I(n)) + kron(inv(G0), Gi)
    ]
    #! format: on
    rhs = [kron(Ri, X'); kron(Ri, I(n))] * vec(Y)
    y, obs = mt_stack(Y)
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [X, X])),
        [RandomEffect(mt_blocks(obs, [I(n), I(n)]), Gi, G0)];
        Rinv = mt_rinv(R0, obs),
    )
    check("mme() LHS = Kronecker form", m.lhs, lhs; atol = 1e-10)

    # 5. Solve MME
    x = lhs \ rhs
    b = x[m.blocks[1]]
    a = reshape(x[m.blocks[2]], :, 2)
    calves.dgv_wwg, calves.dgv_pwg = a[:, 1], a[:, 2]

    # 6. For comparison: pedigree-based multi-trait BLUP of Example 6.1 (same
    #    records, A of the 8-animal pedigree), EBV of calves 4–8
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    Z = incidence(calves.calf, nrow(ped))
    #! format: off
    xa = [
        kron(Ri, X'X)  kron(Ri, X'Z)
        kron(Ri, Z'X)  kron(Ri, Z'Z) + kron(inv(G0), ainv(ped))
    ] \ ([kron(Ri, X'); kron(Ri, Z')] * vec(Y))
    #! format: on
    ea = reshape(xa[5:end], :, 2)[calves.calf, :]
    calves.ebv_wwg, calves.ebv_pwg = ea[:, 1], ea[:, 2]
    check(
        "pedigree EBV = Example 6.1 (WWG, calves 4–8)",
        ea[:, 1],
        [-0.010, -0.270, 0.276, -0.316, 0.244],
    )
    println("\nCalves: genomic (DGV) and pedigree-based (EBV, Example 6.1) values:")
    show_table(calves)
    check("sex (WWG, PWG)", b, [4.323, 3.416, 6.753, 5.921])
    check(
        "calves 4–8",
        a,
        [
            0.081 0.108;
            -0.212 -0.419;
            0.181 0.377;
            -0.209 -0.302;
            0.159 0.235
        ],
    )

    (; b, a, calves)
end
