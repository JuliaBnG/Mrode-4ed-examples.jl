"""
    ex_11_8() -> NamedTuple

Example 11.8: Haplotype model with pseudo-SNPs from phased haplotypes
(Mrode & Pocrnic, 2023, Section 11.6; Calus et al., 2008).

# Model Formulation
When multi-marker linkage disequilibrium provides extra information beyond single SNPs,
phased chromosomal segments (haplotypes) can be grouped into blocks and each block treated
as a multiallelic locus whose haplotype alleles are coded as biallelic pseudo-SNPs:
```math
y = \\mathbf{1}\\mu + a_h + e
```
where:
- Haplotypes across 9 SNPs are partitioned into 3 consecutive blocks of 3 SNPs.
-  Each unique haplotype allele in a block is treated as a pseudo-SNP with dosage 0, 1, or 2
  (`pseudo_snps`).
-  A haplotype-based genomic relationship matrix ``G_h`` is constructed from centered
  pseudo-SNPs.
- Solves GBLUP using both the standard SNP-based ``G`` and the haplotype-based ``G_h``.

# Key Concepts Illustrated
1. **Pseudo-SNP coding** written out: per block, the distinct haplotypes are
   the alleles and the dosage is the number of strands carrying each; compared
   with `pseudo_snps`. ``G_h = Z_hZ_h'/2\\sum pq`` from the centred pseudo-SNPs.
2. **GBLUP** with ``W = I`` (one record per animal), MME
   ``[n\\ \\mathbf{1}';\\ \\mathbf{1}\\ I + \\lambda K^{-1}]`` for each ``K``; written out and compared
   with `mme`. The DGV of all three models are columns of a table of animals.

# Notes on Text Discrepancies
1.  The book prints animal 5's block-2 pseudo-SNPs as `01000`, although its haplotypes `001`
   and `000` give `01001`. The book's printed ``G_h``, ``2\\sum pq = 4.42``, and haplotype
   GBLUP solutions were computed from the misprinted row. Both the printed version and the
   corrected version are checked.
2.  The book states a correlation of 0.999 between SNP and haplotype DGVs; the correlation
   from the book's own printed solutions is 0.992.

# Returns
A `NamedTuple` with fields:
- `G`: Single-SNP genomic relationship matrix.
- `Gh`: Corrected haplotype-based genomic relationship matrix.
-  `sol`: GBLUP solutions under SNP model, book's pseudo-SNP model, and corrected haplotype
  model.
- `animals`: Table of the animals with records and the DGV of the three models.
"""
function ex_11_8()
    println("Example 11.8: SNP- and haplotype-based GBLUP")

    # 1. Phased haplotypes for animals 1 to 5 (maternal and paternal strands, 9 SNPs)
    H = [
        1 0 1 0 1 0 1 1 0;
        1 0 0 0 0 0 1 1 0
        0 1 1 1 1 1 0 1 0;
        0 1 0 0 0 0 0 1 1
        1 1 1 0 1 1 0 0 1;
        1 0 0 0 0 1 0 0 0
        0 1 1 1 1 1 0 1 0;
        1 0 0 0 0 0 1 1 0
        1 0 0 0 0 1 0 0 1;
        0 0 0 0 0 0 0 1 1
    ]
    M = H[1:2:end, :] + H[2:2:end, :]

    # 2. Pseudo-SNPs for 3 blocks of 3 SNPs: in each block, every distinct
    #    haplotype is an allele, and an animal's dosage of it is the number of
    #    its two strands carrying it
    blocks = [1:3, 4:6, 7:9]
    P, alleles = pseudo_snps(H, blocks)
    Pe = reduce(hcat, map(blocks) do b
        hb = collect(eachrow(H[:, b]))
        al = unique(hb)                         # alleles in order of appearance
        [count(==(a), hb[(2i-1):2i]) for i = 1:size(M, 1), a in al]
    end)
    check("pseudo-SNPs = pseudo_snps()", Pe, P; atol = 0)

    # Note on book typo:
    # Animal 5's block-2 pseudo-SNPs were printed as 01000 instead of 01001.
    Pbook = [
        1 1 0 0 0 0 1 1 0 0 0 2 0 0 0 0;
        0 0 1 1 0 0 0 1 1 0 0 0 1 1 0 0;
        0 1 0 0 1 0 0 0 0 1 1 0 0 0 1 1;
        0 1 1 0 0 0 0 1 1 0 0 1 1 0 0 0;
        0 1 0 0 0 1 0 1 0 0 0 0 0 1 1 0
    ]
    check(
        "pseudo-SNPs (book's typo corrected)",
        P,
        Pbook + [zeros(Int, 4, 16); zeros(Int, 1, 10) 1 zeros(Int, 1, 5)];
        atol = 0,
    )
    println("  haplotype alleles of block 1: ", join(join.(alleles[1]), ", "))

    # 3. Construct G matrices (SNP-based, book haplotype, corrected haplotype)
    _, ps, ks = center_genotypes(M)
    _, pb, kb = center_genotypes(Pbook)
    _, ph, kh = center_genotypes(P)
    G = grm(Matrix{Int8}(M'), ps)
    Gb = grm(Matrix{Int8}(Pbook'), pb)
    Gh = grm(Matrix{Int8}(P'), ph)
    Zh = P .- 2ph'
    check("Gₕ = ZₕZₕ'/2Σpq (corrected pseudo-SNPs)", Gh, Zh * Zh' / kh; atol = 1e-12)

    check("2Σpq for SNPs and the book's pseudo-SNPs", [ks, kb], [4.06, 4.42]; atol = 0.005)
    check("G, last row", G[5, :], [-0.424, -0.227, 0.167, -0.522, 1.005])
    check(
        "Gₕ from the book's pseudo-SNPs, last row",
        Gb[5, :],
        [-0.235, -0.100, 0.036, -0.281, 0.579],
    )

    # 4. GBLUP with each relationship matrix. Every animal has one record, so
    #    W = I and the MME are [n 1'; 1 I + λK⁻¹], λ = σ²e/σ²a = 2
    animals = DataFrame(animal = 1:5, wwg = [4.5, 2.9, 3.9, 3.5, 5.0])
    y = animals.wwg
    X = ones(5, 1)
    λ = 40.0 / 20.0
    sol = map((("SNP", G), ("book haplotype", Gb), ("haplotype", Gh))) do (lab, K)
        Ki = inv(K + 0.01I)
        lhs = [X'X X'; X I + λ * Ki]
        m = mme(y, X, [RandomEffect(I(5), Ki, 20.0)]; σ²e = 40.0)
        check("$lab: mme() LHS = [n 1'; 1 I + λK⁻¹]", m.lhs, lhs; atol = 1e-10)
        lhs \ [X'y; y]
    end
    animals.dgv_snp = sol[1][2:end]
    animals.dgv_hap_book = sol[2][2:end]
    animals.dgv_hap = sol[3][2:end]
    println("Means: ", round.(first.(sol); digits = 3))
    show_table(animals)
    check("SNP-based GBLUP", sol[1], [3.960, 0.253, -0.474, 0.057, -0.224, 0.388])
    check(
        "haplotype-based GBLUP (book's pseudo-SNPs)",
        sol[2],
        [3.960, 0.235, -0.380, 0.079, -0.196, 0.262],
    )

    # Note on book correlation claim: 0.999 reported vs 0.992 calculated
    check(
        "correlation of DGVs (book's solutions)",
        cor(sol[1][2:end], sol[2][2:end]),
        cor([0.253, -0.474, 0.057, -0.224, 0.388], [0.235, -0.380, 0.079, -0.196, 0.262]),
    )
    @printf(
        "  with corrected pseudo-SNPs: 2Σpq = %.2f, cor(SNP, haplotype DGV) = %.3f\n",
        kh,
        cor(sol[1][2:end], sol[3][2:end])
    )

    (; G, Gh, sol, animals)
end
