"""
    ex_11_3() -> NamedTuple

Examples 11.3–11.5: GBLUP, back-solving SNP effects, selection index form, and
cross-validation (Mrode & Pocrnic, 2023, Sections 11.4.2, 11.4.4–11.4.5 and 11.9; VanRaden,
2008).

# Model Formulation and Equivalence
GBLUP fits animal genetic effects using the genomic relationship matrix ``G``:
```math
y = \\mathbf{1}\\mu + W a + e
```
where ``\\mathrm{var}(a) = G \\sigma_a^2``, with VanRaden (2008) method 1:
```math
G = \\frac{Z Z'}{2 \\sum p_j (1 - p_j)}
```
with ``Z = M - 2p`` (centered genotypes).

# Key Concepts Illustrated
1. **VanRaden G Matrix Construction**: Built via `grm(M', p)`, checked against
   ``ZZ'/k``.
2. **GBLUP MME** ``[X'X\\ X'W;\\ W'X\\ W'W + \\lambda G^{-1}]``, written out and
   compared with `mme`. With 10 SNPs, ``G`` (14 × 14) has rank 10 and is
   singular, so ``G + 0.01I`` is inverted.
3. **Equivalence of GBLUP and SNP-BLUP**:
   GBLUP animal solutions ``\\hat{a}`` equal ``Z \\hat{g}`` from SNP-BLUP, here up to
   the regularization of ``G`` (within 0.01). The selection index, which needs
   no inverse of ``G``, reproduces the SNP-BLUP values.
4. **Back-solving Marker Effects from GBLUP** (Example 11.4; Strandén & Garrick, 2009):
   ```math
   \\hat{g} = \\frac{1}{k} Z' G^{-1} \\hat{a}
   ```
5. **Selection Index Form of GBLUP** (Example 11.5):
   Prediction for candidates without solving the full MME:
   ``\\hat{a}_2 = G_{21} (G_{11} + \\alpha I)^{-1} (y_1 - \\mathbf{1}\\hat{\\mu})``.
   Back-solving and the index are written out and compared with
   `snp_from_gblup` and `gblup_index`.
6.  **Realized Accuracy vs Theoretical Reliability** (Section 11.9): Validation correlation
   ``r = \\mathrm{cor}(\\hat{a}_{\\mathrm{cand}}, y_{\\mathrm{cand}})`` compared with mean
   theoretical reliability.

# Returns
A `NamedTuple` with fields:
- `G`: Genomic relationship matrix (14 × 14).
- `a`: Vector of GBLUP DGV for all 14 bulls.
- `g`: Vector of back-solved SNP effects.
- `dgv`: Direct genomic values from the selection index formulation.
- `rel`: Theoretical reliabilities from selection index.
- `bulls`: The bulls with GBLUP, SNP-BLUP and selection-index DGV and reliabilities added.
- `snps`: Table of SNPs with back-solved and SNP-BLUP effects.
"""
function ex_11_3()
    println("Examples 11.3–11.5: GBLUP and equivalent models")

    # 1. Bulls (one row per genotyped bull, in the row order of M), and the
    #    VanRaden G matrix (Example 11.3)
    (; bulls, M, σ²a, σ²e) = data_11()
    isref = bulls.set .== "reference"
    iscand = .!isref
    λ = σ²e / σ²a
    Z, p, k = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)                  # VanRaden (2008) method 1
    check("G = ZZ'/2Σpq", G, Z * Z' / k; atol = 1e-12)
    check(
        "G, first column",
        G[:, 1],
        [
            1.472,
            -0.446,
            0.988,
            0.059,
            0.685,
            -0.163,
            -0.708,
            -0.547,
            0.887,
            -0.789,
            -0.203,
            -0.143,
            -0.829,
            -0.264,
        ],
    )
    check(
        "G, last row",
        G[14, :],
        [
            -0.264,
            0.362,
            -0.466,
            -0.264,
            -0.486,
            -0.203,
            0.100,
            -0.304,
            -0.284,
            0.584,
            -0.526,
            0.382,
            0.261,
            1.109,
        ],
    )

    # 2. GBLUP MME (R = Iσ²e) for the reference bulls, with breeding values of
    #    all 14 bulls: [X'X X'W; W'X W'W + λG⁻¹]. With 10 SNPs, G (14 × 14) has
    #    rank at most 10, so it is regularized with 0.01 I before inversion.
    @printf("  rank(G) = %d of %d\n", rank(G), size(G, 1))
    Gi = inv(G + 0.01I)
    y = bulls.dyd[isref]
    X = ones(count(isref), 1)
    W = incidence(findall(isref), nrow(bulls))  # records → rows of G
    #! format: off
    lhs = [
        X'X  X'W
        W'X  W'W + λ * Gi
    ]
    #! format: on
    rhs = [X'y; W'y]
    m = mme(y, X, [RandomEffect(W, Gi, σ²a)]; σ²e)
    check("mme() LHS = [X'X X'W; W'X W'W + λG⁻¹]", m.lhs, lhs; atol = 1e-10)
    x = lhs \ rhs
    b = x[m.blocks[1]]
    a = x[m.blocks[2]]
    bulls.gblup = a
    check("GBLUP mean", b, [9.944])
    check(
        "GBLUP DGV",
        a,
        [
            0.069,
            0.116,
            0.049,
            0.260,
            -0.500,
            -0.359,
            0.146,
            -0.231,
            0.028,
            0.115,
            -0.240,
            0.143,
            0.054,
            0.353,
        ],
    )

    #    Equivalence with SNP-BLUP (Example 11.2): â = Zĝ, up to the 0.01 I
    #    regularization of G
    Zr = Z[isref, :]
    xs = [X'X X'Zr; Zr'X Zr'Zr + λ * k * I] \ [X'y; Zr'y]
    bulls.snpblup = Z * xs[2:end]
    check("GBLUP ≈ SNP-BLUP DGV (G regularized)", a, bulls.snpblup; atol = 0.01)

    # 3. Example 11.4: Back-solve SNP effects from GBLUP solutions,
    #    ĝ = Z'G⁻¹â / k (Strandén & Garrick, 2009)
    g = Z' * Gi * a / k
    check("ĝ = Z'G⁻¹â/k (snp_from_gblup)", g, snp_from_gblup(Z, Gi, a; k); atol = 1e-12)
    snps = DataFrame(snp = 1:10, p = p, g_backsolved = g, g_snpblup = xs[2:end])
    check(
        "SNP effects back-solved from GBLUP",
        g,
        [0.087, -0.311, 0.262, -0.080, 0.110, 0.139, 0.000, 0.001, -0.061, -0.016],
    )

    # 4. Example 11.5: Selection index form of GBLUP, without the MME:
    #    â = G₍·,r₎ (G₍r,r₎ + λI)⁻¹ (y - 1μ̂), reliability bᵢᵢ/gᵢᵢ with
    #    B = G₍·,r₎ (G₍r,r₎ + λI)⁻¹ G₍r,·₎
    yc = y .- 9.944                             # the book's mean
    V = G[isref, isref] + λ * I
    dgv = G[:, isref] * (V \ yc)
    rel = diag(G[:, isref] * (V \ G[isref, :])) ./ diag(G)
    check(
        "selection index = gblup_index()",
        [dgv rel],
        hcat(gblup_index(G, findall(isref), yc, λ)...);
        atol = 1e-12,
    )
    bulls.index_dgv, bulls.rel = dgv, rel
    check(
        "selection-index DGV",
        dgv,
        [
            0.070,
            0.111,
            0.045,
            0.253,
            -0.495,
            -0.357,
            0.146,
            -0.225,
            0.028,
            0.115,
            -0.240,
            0.143,
            0.054,
            0.353,
        ],
    )

    # 5. Section 11.9: Realized accuracy in candidates vs theoretical reliability
    r = cor(a[iscand], bulls.dyd[iscand])
    check("realized accuracy, reliability", [r, r^2], [0.49, 0.24]; atol = 6e-3)
    @printf("  mean theoretical reliability of candidates: %.3f\n", mean(rel[iscand]))

    println("\nSNP effects back-solved from GBLUP, and from SNP-BLUP:")
    show_table(snps)
    println("\nBulls: GBLUP, SNP-BLUP and selection-index DGV, and reliabilities:")
    show_table(bulls)

    (; G, a, g, dgv, rel, bulls, snps)
end
