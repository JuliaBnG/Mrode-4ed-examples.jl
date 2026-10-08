"""
    ex_11_6() -> NamedTuple

Example 11.6: Base-population allele frequencies and imputed ancestral gene content
(Mrode & Pocrnic, 2023, Section 11.4.6; Gengler et al., 2007; Garcia-Baccino et al., 2017).

# Model Formulation
When only a subset of animals (e.g. bulls 13–26) are genotyped, gene content (allele dosages
0, 1, or 2) for ungenotyped ancestors (animals 1–12) can be imputed using pedigree
relationships:
```math
q_i = \\hat{\\mu} + \\hat{d}_i
```
where:
-  ``\\mu = 2 p_{\\mathrm{base}}`` is the expected gene content in the unselected base
  population.
-  ``d`` is the vector of animal genetic deviations in gene content, predicted via BLUP
  using ``A^{-1}``.
-  Imputed gene contents can be rounded to integer genotypes (0, 1, 2) or used as expected
  dosage probabilities.
-  GLS estimator (Garcia-Baccino et al., 2017):
  ``\\hat{\\mu} = \\frac{\\mathbf{1}' A_{22}^{-1} M_{:, 1}}{\\mathbf{1}' A_{22}^{-1}
  \\mathbf{1}}`` (mean gene content of SNP 1; the book reports 0.575).

# Returns
A `NamedTuple` with fields:
- `p`: Base allele frequencies for all 10 SNPs.
- `μ`: Base population gene content expectations (``2 p``).
- `d`: Matrix of BLUP gene content deviations for all 26 animals across 10 SNPs.
"""
function ex_11_6()
    println("Example 11.6: base allele frequencies and imputed gene content")

    # 1. Load data and setup pedigree
    (; ped, M, id) = data_11()
    Ai = ainv(ped)

    # 2. Linear method (Gengler et al., 2007) for base frequencies and gene content BLUPs
    p, μ, d = base_allele_frequencies(M, Ai, id; λ = 0.01)
    q = μ[1] .+ d[:, 1]                           # predicted gene content for SNP 1

    show_table(animal = 1:26, BLUP = d[:, 1], predicted = q, rounded = round.(Int, q))
    check("mean of SNP 1 (= 2p̂)", μ[1], 0.597)
    check(
        "BLUPs of SNP 1",
        d[:, 1],
        [
            0.695,
            -0.523,
            0.695,
            -0.198,
            -0.523,
            0.140,
            -0.518,
            -0.518,
            -0.779,
            0.140,
            -0.518,
            0.140,
            1.387,
            0.380,
            0.397,
            -0.586,
            -0.586,
            0.400,
            -0.589,
            -0.589,
            1.389,
            -0.587,
            -0.587,
            0.400,
            -0.587,
            0.400,
        ];
        atol = 1.5e-3,
    )
    check(
        "rounded genotypes of genotyped bulls = true",
        round.(Int, q[id]),
        M[:, 1];
        atol = 0,
    )

    # 3. Compare with GLS estimator (Garcia-Baccino et al., 2017)
    # Note: The book reports 0.575; the GLS estimator with (A₂₂)⁻¹ gives ~0.590, close to μ̂ = 0.597
    o = ones(14)
    gls(K) = (o' * K * M[:, 1]) / (o' * K * o)
    @printf(
        "  GLS mean: with (A⁻¹)₂₂ %.3f, with (A₂₂)⁻¹ %.3f (book 0.575)\n",
        gls(Matrix(Ai[id, id])),
        gls(inv(nrm(ped, id)))
    )
    println("  base allele frequencies of all SNPs: ", round.(p; digits = 3))

    (; p, μ, d)
end
