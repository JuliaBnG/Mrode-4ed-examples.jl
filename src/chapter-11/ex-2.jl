"""
    ex_11_2() -> NamedTuple

Example 11.2: SNP-BLUP with random marker effects (unweighted and EDC-weighted)
(Mrode & Pocrnic, 2023, Section 11.4.1; Meuwissen et al., 2001).

# Model Formulation
SNP-BLUP assumes all markers have infinitesimal effects drawn from a common normal
distribution:
```math
y = \\mathbf{1}\\mu + Z g + e
```
where:
-  ``g_j \\sim N(0, \\sigma_g^2)`` with
  ``\\sigma_g^2 = \\frac{\\sigma_a^2}{2 \\sum p_j(1 - p_j)} = \\frac{\\sigma_a^2}{k}``.
-  Variance ratio for each marker equation:
  ``\\alpha_{\\mathrm{SNP}} = \\frac{\\sigma_e^2}{\\sigma_g^2} = k
  \\frac{\\sigma_e^2}{\\sigma_a^2}``.
- Direct Genomic Values (DGV) for reference animals and selection candidates:
  ``\\mathrm{DGV} = Z \\hat{g}``.

# Key Concepts Illustrated
1. **Tables**: SNP effects in a table of SNPs (with allele frequencies), DGV
   as columns of the bull table; the genotypes stay a matrix.
2. **MME in the general form**,
   ``[\\mathbf{1}'R^{-1}\\mathbf{1}\\ \\mathbf{1}'R^{-1}Z;\\ Z'R^{-1}\\mathbf{1}\\ Z'R^{-1}Z + I/\\sigma_g^2]``,
   unweighted and EDC-weighted as in Example 11.1; written out and compared
   with `mme`.
3. **Monomorphic SNPs**: SNPs 7 and 8 do not segregate among the reference
   bulls; their centred columns are confounded with the mean, so their
   effects are exactly zero.

# Returns
A `NamedTuple` with fields:
- `snps`: Table of SNPs with allele frequencies and both sets of effects.
- `bulls`: The bulls with the DGV of both analyses added.
- `x`: Unweighted SNP-BLUP solutions (mean and 10 SNP effects).
- `xw`: EDC-weighted SNP-BLUP solutions.
"""
function ex_11_2()
    println("Example 11.2: SNP-BLUP")

    # 1. Bulls (one row per genotyped bull, in the row order of M) and centred
    #    genotypes Z = M - 2p
    (; bulls, M, σ²a, σ²e) = data_11()
    Z, p, k = center_genotypes(M)
    σ²g = σ²a / k                               # variance of each SNP effect
    check("2Σpq and α", [k, σ²e / σ²g], [3.5383, 24.598]; atol = 2e-3)

    # 2. MME in the general form for the reference bulls,
    #    [1'R⁻¹1 1'R⁻¹Z; Z'R⁻¹1 Z'R⁻¹Z + I/σ²g], unweighted (R⁻¹ = I/σ²e) and
    #    EDC-weighted (R⁻¹ = diag(EDC/σ²e)); compared with mme()
    isref = bulls.set .== "reference"
    ref = bulls[isref, :]
    y = ref.dyd
    Zr = Z[isref, :]
    X = ones(nrow(ref), 1)                      # overall mean
    fits = [("unweighted", fill(1 / σ²e, nrow(ref))), ("weighted", ref.edc ./ σ²e)]
    x, xw = map(fits) do (lab, rinv)
        Ri = Diagonal(rinv)
        lhs = [
            X'Ri*X   X'Ri*Zr
            Zr'Ri*X  Zr'Ri*Zr + I / σ²g
        ]
        rhs = [X'Ri*y; Zr'Ri*y]
        m = mme(y, X, [RandomEffect(Zr, I, σ²g)]; Rinv = rinv)
        check("$lab: mme(; Rinv) LHS = general form", m.lhs, lhs; atol = 1e-12)
        lhs \ rhs
    end

    # 3. SNP effects per SNP. SNPs 7 and 8 are monomorphic in the reference bulls:
    #    their centred columns are constant, confounded with the unpenalized mean,
    #    and their effects are shrunk to exactly zero.
    snps = DataFrame(snp = 1:10, p = p, g_unweighted = x[2:end], g_weighted = xw[2:end])
    mono = [allequal(c) for c in eachcol(M[isref, :])]
    check("SNPs monomorphic in the reference", findall(mono), [7, 8]; atol = 0)
    check("their effects are zero", [x[2:end][mono]; xw[2:end][mono]], zeros(4); atol = 1e-10)
    println("\nMean: unweighted ", round(x[1]; digits = 3), ", weighted ", round(xw[1]; digits = 3))
    show_table(snps)

    check(
        "unweighted: mean, SNP effects",
        x,
        [9.944, 0.087, -0.311, 0.262, -0.080, 0.110, 0.139, 0, 0, -0.061, -0.016],
    )
    check(
        "weighted: mean, SNP effects",
        xw,
        [11.876, -0.633, -3.041, 3.069, -1.267, 2.600, 4.447, 0, 0, -3.240, 1.883],
    )

    # 4. DGV of reference bulls and candidates: DGV = Z ĝ
    bulls.dgv_unweighted = Z * x[2:end]
    bulls.dgv_weighted = Z * xw[2:end]
    check(
        "unweighted DGV (reference, candidates)",
        bulls.dgv_unweighted,
        [
            0.070,
            0.111,
            0.045,
            0.253,
            -0.495,
            -0.357,
            0.145,
            -0.224,
            0.027,
            0.114,
            -0.240,
            0.143,
            0.054,
            0.354,
        ],
    )
    check(
        "weighted DGV (reference, candidates)",
        bulls.dgv_weighted,
        [
            -2.651,
            1.307,
            0.611,
            1.007,
            -5.693,
            -4.358,
            0.502,
            -5.718,
            -0.006,
            6.513,
            -3.835,
            2.701,
            3.273,
            6.350,
        ],
    )
    show_table(bulls)

    (; snps, bulls, x, xw)
end
