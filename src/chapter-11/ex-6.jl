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
  dosage probabilities. The linear predictor is not bounded to ``[0, 2]`` (animal 9 gets
  ``-0.18`` for SNP 1), so rounded values are clamped.
-  GLS estimator (Garcia-Baccino et al., 2017):
  ``\\hat{\\mu} = \\frac{\\mathbf{1}' A_{22}^{-1} M_{:, 1}}{\\mathbf{1}' A_{22}^{-1}
  \\mathbf{1}}`` (mean gene content of SNP 1; the book reports 0.575).

# Key Concepts Illustrated
1. **Every SNP is a trait with the same MME**:
   ``[X'X\\ X'W;\\ W'X\\ W'W + \\lambda A^{-1}]\\,[\\mu';\\ D] = [X'M;\\ W'M]`` is solved for
   all SNPs at once with a matrix right-hand side; compared with
   `base_allele_frequencies`.
2. **Tables**: observed and predicted gene content of SNP 1 as columns of
   the pedigree; base allele frequencies next to the frequencies among the
   genotyped bulls; imputed genotypes of the ungenotyped ancestors.

# Returns
A `NamedTuple` with fields:
- `p`: Base allele frequencies for all 10 SNPs.
- `μ`: Base population gene content expectations (``2 p``).
- `d`: Matrix of BLUP gene content deviations for all 26 animals across 10 SNPs.
- `ped`: The pedigree with `genotyped`, `snp1` (observed), `d1`, `q1` and `q1_rounded` added.
- `snps`: Base and genotyped-sample allele frequencies.
- `Q`: Predicted gene content of all animals (26 × 10).
"""
function ex_11_6()
    println("Example 11.6: base allele frequencies and imputed gene content")

    # 1. Pedigree (one row per animal; 1–12 ungenotyped) and genotypes M of
    #    bulls 13–26
    (; ped, bulls, M) = data_11()
    id = bulls.id
    Ai = ainv(ped)

    # 2. Linear method (Gengler et al., 2007): each SNP is a trait,
    #    q = 1μ + Wd + e with var(d) = Aσ²d and λ = σ²e/σ²d = 0.01. The MME are
    #    the same for all SNPs; only the right-hand side differs, so all 10 SNPs
    #    are solved at once with a matrix right-hand side [X'M; W'M].
    λ = 0.01
    X = ones(length(id), 1)
    W = incidence(id, nrow(ped))
    #! format: off
    lhs = [
        X'X  X'W
        W'X  W'W + λ * Ai
    ]
    #! format: on
    S = lhs \ [X'M; W'M]                       # (1 + 26) × 10: μ' on top of D
    μ, d = S[1, :], S[2:end, :]
    p = μ / 2
    check(
        "explicit = base_allele_frequencies()",
        [p; vec(d)],
        (r = base_allele_frequencies(M, Ai, id; λ); [r[1]; vec(r[3])]);
        atol = 1e-10,
    )

    # 3. Predicted gene content of every animal and SNP, Q = 1μ' + D; for the
    #    genotyped bulls it reproduces their genotypes. The linear predictor is
    #    not bounded to [0, 2] (animal 9: -0.18), so genotypes are clamped.
    Q = μ' .+ d
    genotype(q) = clamp(round(Int, q), 0, 2)
    q = Q[:, 1]                                 # SNP 1
    ped.genotyped = in(Set(id)).(ped.id)
    ped.snp1 = Vector{Union{Missing,Int}}(missing, nrow(ped))
    ped.snp1[id] = M[:, 1]
    ped.d1, ped.q1, ped.q1_rounded = d[:, 1], q, genotype.(q)
    snps = DataFrame(snp = 1:10, p_base = p, p_genotyped = vec(mean(M; dims = 1)) / 2)
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
    check(
        "rounded gene content of genotyped bulls = M (all SNPs)",
        genotype.(Q[id, :]),
        M;
        atol = 0,
    )

    println("\nPedigree with observed and predicted gene content of SNP 1:")
    show_table(ped)
    println("\nBase allele frequencies and frequencies among the genotyped bulls:")
    show_table(snps)
    println("\nImputed genotypes of the ungenotyped animals 1–12 (rounded Q):")
    show(stdout, "text/plain", genotype.(Q[.!ped.genotyped, :]))
    println()

    (; p, μ, d, ped, snps, Q)
end
