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

# Returns
A `NamedTuple` with fields:
- `b`: Fixed sex effects for WWG and PWG.
- `a`: Direct genomic values (DGVs) for calves 4 to 8 for both traits.
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
    sex = ["M", "F", "F", "M", "M"]
    Y = [4.5 6.8; 2.9 5.0; 3.9 6.8; 3.5 6.0; 5.0 7.5]

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

    # 4. Multi-trait GBLUP MME setup
    X, _ = incidence(sex; levels = ["M", "F"])
    y, obs = mt_stack(Y)
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [X, X])),
        [RandomEffect(mt_blocks(obs, [I(5), I(5)]), inv(G + 0.01I), G0)];
        Rinv = mt_rinv(R0, obs),
    )

    # 5. Solve MME
    b, (a,) = solutions(m, solve_mme(m))

    show_table(calf = 4:8, WWG = a[:, 1], PWG = a[:, 2])
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

    (; b, a)
end
