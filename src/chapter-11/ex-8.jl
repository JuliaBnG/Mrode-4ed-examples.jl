# Example 11.8 – haplotype model: pseudo-SNPs from phased haplotypes and
# GBLUP with the haplotype-based relationship matrix.

function ex_11_8()
    println("Example 11.8: SNP- and haplotype-based GBLUP")
    # maternal (f) and paternal (m) haplotypes of animals 1–5, nine SNPs
    H = [1 0 1 0 1 0 1 1 0; 1 0 0 0 0 0 1 1 0
         0 1 1 1 1 1 0 1 0; 0 1 0 0 0 0 0 1 1
         1 1 1 0 1 1 0 0 1; 1 0 0 0 0 1 0 0 0
         0 1 1 1 1 1 0 1 0; 1 0 0 0 0 0 1 1 0
         1 0 0 0 0 1 0 0 1; 0 0 0 0 0 0 0 1 1]
    M = H[1:2:end, :] + H[2:2:end, :]
    P, alleles = pseudo_snps(H, [1:3, 4:6, 7:9])
    # The book prints animal 5's block-2 pseudo-SNPs as 01000 although its
    # haplotypes 001 and 000 give 01001; its Gₕ (2Σpq = 4.42) and the
    # haplotype GBLUP solutions were computed from that printed row.
    Pbook = [1 1 0 0 0 0 1 1 0 0 0 2 0 0 0 0; 0 0 1 1 0 0 0 1 1 0 0 0 1 1 0 0;
             0 1 0 0 1 0 0 0 0 1 1 0 0 0 1 1; 0 1 1 0 0 0 0 1 1 0 0 1 1 0 0 0;
             0 1 0 0 0 1 0 1 0 0 0 0 0 1 1 0]
    check("pseudo-SNPs (book's typo corrected)", P, Pbook + [zeros(Int, 4, 16); zeros(Int, 1, 10) 1 zeros(Int, 1, 5)];
          atol = 0)
    println("  haplotype alleles of block 1: ", join(join.(alleles[1]), ", "))
    _, ps, ks = center_genotypes(M)
    _, pb, kb = center_genotypes(Pbook)
    _, ph, kh = center_genotypes(P)
    G = grm(Matrix{Int8}(M'), ps)
    Gb = grm(Matrix{Int8}(Pbook'), pb)
    Gh = grm(Matrix{Int8}(P'), ph)
    check("2Σpq for SNPs and the book's pseudo-SNPs", [ks, kb], [4.06, 4.42]; atol = 0.005)
    check("G, last row", G[5, :], [-0.424, -0.227, 0.167, -0.522, 1.005])
    check("Gₕ from the book's pseudo-SNPs, last row", Gb[5, :],
          [-0.235, -0.100, 0.036, -0.281, 0.579])

    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]
    sol = map((G, Gb, Gh)) do K
        m = mme(wwg, ones(5, 1), [RandomEffect(I(5), inv(K + 0.01I), 20.0)]; σ²e = 40.0)
        solve_mme(m)
    end
    show_table(effect = ["mean", string.(1:5)...], SNP = sol[1], haplotype_book = sol[2],
               haplotype_corrected = sol[3])
    check("SNP-based GBLUP", sol[1], [3.960, 0.253, -0.474, 0.057, -0.224, 0.388])
    check("haplotype-based GBLUP (book's pseudo-SNPs)", sol[2],
          [3.960, 0.235, -0.380, 0.079, -0.196, 0.262])
    # the book states a correlation of 0.999; its own solutions give 0.992
    check("correlation of DGVs (book's solutions)", cor(sol[1][2:end], sol[2][2:end]),
          cor([0.253, -0.474, 0.057, -0.224, 0.388], [0.235, -0.380, 0.079, -0.196, 0.262]))
    @printf("  with corrected pseudo-SNPs: 2Σpq = %.2f, cor(SNP, haplotype DGV) = %.3f\n",
            kh, cor(sol[1][2:end], sol[3][2:end]))
    (; G, Gh, sol)
end
