"""
    ex_11_7() -> NamedTuple

Example 11.7: Genomic prediction with a residual polygenic effect
(Mrode & Pocrnic, 2023, Section 11.5).

# Model Formulation
When markers do not capture all genetic variation (e.g. incomplete LD or rare variants),
a residual polygenic effect ``u`` is included alongside genomic effects:
```math
a = u + a_g
```
where:
-  ``u \\sim N(0, A \\sigma_u^2)`` is the residual polygenic effect (here 10% of
  ``\\sigma_a^2``: ``\\sigma_u^2 = 0.1 \\sigma_a^2``).
-  ``a_g`` is the genomic component (here 90% of ``\\sigma_a^2``:
  ``\\sigma_g^2 = 0.9 \\sigma_a^2``).
- Two equivalent models are formulated:
  1.  **SNP-BLUP + Polygenic** (Eqn 11.14): Fits random SNP effects ``g`` and random animal
     effect ``u``.
  2.  **GBLUP + Polygenic** (Eqn 11.16): Fits two random animal effects
     (``u \\sim N(0, A \\sigma_u^2)`` and ``a_g \\sim N(0, G \\sigma_g^2)``).

# Notes on Text Discrepancies
The book's Table 11.4 headings label the results as "DGV", but the printed values represent
``\\mathrm{DGV} + \\hat{u}`` (genomic breeding value including the polygenic effect).
Furthermore, the polygenic component was computed using a 5-generation pedigree relationship
matrix ``A`` printed in Example 11.3 rather than the 2-generation pedigree of Table 11.1.

# Returns
A `NamedTuple` with fields:
- `bs`: Fixed mean from SNP-BLUP model.
- `us`: Vector of polygenic solutions from SNP-BLUP.
- `gs`: Vector of SNP effects from SNP-BLUP.
- `bg`: Fixed mean from GBLUP model.
- `ug`: Vector of polygenic solutions from GBLUP.
- `ag`: Vector of genomic solutions from GBLUP.
"""
function ex_11_7()
    println("Example 11.7: models with polygenic effects")

    # 1. Load data and setup variance components
    (; M, dyd, ref, σ²a, σ²e) = data_11()
    Z, p, k = center_genotypes(M)
    σ²u = 0.1 * σ²a                             # 10% residual polygenic variance
    σ²g = σ²a - σ²u                             # 90% genomic variance

    # 2. 5-generation pedigree relationship matrix from Example 11.3 (as used in the book)
    A = [
        1.008 0.033 0.545 0.288 0.285 0.047 0.033 0.033 0.099 0.046 0.096 0.041 0.033 0.035
        0.033 1.037 0.021 0.021 0.031 0.580 0.613 0.613 0.031 0.586 0.569 0.574 0.548 0.588
        0.545 0.021 1.041 0.536 0.541 0.036 0.021 0.021 0.082 0.032 0.067 0.027 0.035 0.023
        0.288 0.021 0.536 1.016 0.293 0.028 0.021 0.021 0.118 0.031 0.043 0.019 0.039 0.024
        0.285 0.031 0.541 0.293 1.020 0.032 0.031 0.031 0.074 0.039 0.047 0.026 0.039 0.039
        0.047 0.580 0.036 0.028 0.032 1.062 0.365 0.365 0.028 0.351 0.329 0.331 0.315 0.337
        0.033 0.613 0.021 0.021 0.031 0.365 1.095 0.613 0.031 0.373 0.357 0.406 0.336 0.376
        0.033 0.613 0.021 0.021 0.031 0.365 0.613 1.095 0.031 0.373 0.357 0.406 0.336 0.376
        0.099 0.031 0.082 0.118 0.074 0.028 0.031 0.031 1.021 0.044 0.042 0.028 0.037 0.036
        0.046 0.586 0.032 0.031 0.039 0.351 0.373 0.373 0.044 1.068 0.338 0.335 0.321 0.347
        0.096 0.569 0.067 0.043 0.047 0.329 0.357 0.357 0.042 0.338 1.050 0.335 0.310 0.341
        0.041 0.574 0.027 0.019 0.026 0.331 0.406 0.406 0.028 0.335 0.335 1.056 0.310 0.348
        0.033 0.548 0.035 0.039 0.039 0.315 0.336 0.336 0.037 0.321 0.310 0.310 1.029 0.325
        0.035 0.588 0.023 0.023 0.039 0.337 0.376 0.376 0.036 0.347 0.341 0.348 0.325 1.070
    ]
    Ai = inv(A)
    W = incidence(ref, 14)
    y = dyd[ref]
    X = ones(8, 1)

    check(
        "α₁, α₂ (SNP-BLUP), α₂ (GBLUP)",
        [σ²e / σ²u, k * σ²e / σ²g, σ²e / σ²g],
        [69.521, 27.332, 7.725];
        atol = 2e-3,
    )

    # 3. Model 1: SNP-BLUP + Polygenic (Eqn 11.14)
    ms = mme(y, X, [RandomEffect(W, Ai, σ²u), RandomEffect(Z[ref, :], I, σ²g / k)]; σ²e)
    bs, (us, gs) = solutions(ms, solve_mme(ms))

    # 4. Model 2: GBLUP + Polygenic (Eqn 11.16)
    G = grm(Matrix{Int8}(M'), p)
    mg = mme(y, X, [RandomEffect(W, Ai, σ²u), RandomEffect(W, inv(G + 0.01I), σ²g)]; σ²e)
    bg, (ug, ag) = solutions(mg, solve_mme(mg))

    show_table(
        bull = 13:26,
        poly_snp = vec(us),
        DGV_snp = Z * vec(gs),
        poly_g = vec(ug),
        DGV_g = vec(ag),
    )
    check(
        "SNP-BLUP: mean, SNP effects",
        [bs; vec(gs)],
        [9.940, 0.078, -0.280, 0.234, -0.075, 0.098, 0.128, 0, 0, -0.054, -0.018],
    )
    check("GBLUP mean", bg, [9.940])
    check(
        "polygenic (both models)",
        [vec(us) vec(ug)],
        repeat(
            [
                0.011,
                -0.007,
                0.043,
                0.076,
                -0.015,
                -0.025,
                -0.021,
                -0.056,
                0.005,
                -0.006,
                -0.004,
                -0.008,
                -0.003,
                -0.006,
            ],
            1,
            2,
        ),
    )

    # Note: Table 11.4 in the book tabulates DGV + polygenic
    check(
        "SNP-BLUP DGV + polygenic",
        Z * vec(gs) + vec(us),
        [
            0.066,
            0.102,
            0.071,
            0.299,
            -0.473,
            -0.343,
            0.115,
            -0.254,
            0.028,
            0.102,
            -0.220,
            0.125,
            0.051,
            0.316,
        ],
    )
    check(
        "GBLUP DGV + polygenic",
        vec(ag) + vec(ug),
        [
            0.064,
            0.106,
            0.074,
            0.305,
            -0.477,
            -0.345,
            0.115,
            -0.260,
            0.029,
            0.102,
            -0.220,
            0.125,
            0.051,
            0.315,
        ],
    )

    (; bs, us = vec(us), gs = vec(gs), bg, ug = vec(ug), ag = vec(ag))
end
