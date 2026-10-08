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

    # 2. Construct pseudo-SNPs for 3 blocks of 3 SNPs
    P, alleles = pseudo_snps(H, [1:3, 4:6, 7:9])

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

    check("2Σpq for SNPs and the book's pseudo-SNPs", [ks, kb], [4.06, 4.42]; atol = 0.005)
    check("G, last row", G[5, :], [-0.424, -0.227, 0.167, -0.522, 1.005])
    check(
        "Gₕ from the book's pseudo-SNPs, last row",
        Gb[5, :],
        [-0.235, -0.100, 0.036, -0.281, 0.579],
    )

    # 4. Fit GBLUP models
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]
    sol = map((G, Gb, Gh)) do K
        m = mme(wwg, ones(5, 1), [RandomEffect(I(5), inv(K + 0.01I), 20.0)]; σ²e = 40.0)
        solve_mme(m)
    end

    show_table(
        effect = ["mean", string.(1:5)...],
        SNP = sol[1],
        haplotype_book = sol[2],
        haplotype_corrected = sol[3],
    )
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

    (; G, Gh, sol)
end
