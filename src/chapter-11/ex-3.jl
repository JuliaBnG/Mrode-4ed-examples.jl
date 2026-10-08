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
1. **VanRaden G Matrix Construction**: Built via `grm(M', p)`.
2. **Equivalence of GBLUP and SNP-BLUP**:
   GBLUP animal solutions ``\\hat{a}`` equal ``Z \\hat{g}`` from SNP-BLUP.
3. **Back-solving Marker Effects from GBLUP** (Example 11.4; Strandén & Garrick, 2009):
   ```math
   \\hat{g} = \\frac{1}{k} Z' G^{-1} \\hat{a}
   ```
4. **Selection Index Form of GBLUP** (Example 11.5):
   Prediction for candidates without solving the full MME:
   ``\\hat{a}_2 = G_{21} (G_{11} + \\alpha I)^{-1} (y_1 - \\mathbf{1}\\hat{\\mu})``.
5.  **Realized Accuracy vs Theoretical Reliability** (Section 11.9): Validation correlation
   ``r = \\mathrm{cor}(\\hat{a}_{\\mathrm{cand}}, y_{\\mathrm{cand}})`` compared with mean
   theoretical reliability.

# Returns
A `NamedTuple` with fields:
- `G`: Genomic relationship matrix (14 × 14).
- `a`: Vector of GBLUP DGV for all 14 bulls.
- `g`: Vector of back-solved SNP effects.
- `dgv`: Direct genomic values from the selection index formulation.
- `rel`: Theoretical reliabilities from selection index.
"""
function ex_11_3()
    println("Examples 11.3–11.5: GBLUP and equivalent models")

    # 1. Load data and construct VanRaden G matrix (Example 11.3)
    (; M, dyd, ref, cand, σ²a, σ²e) = data_11()
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

    # 2. Fit GBLUP MME for reference bulls (regularized with 0.01 I)
    Gi = inv(G + 0.01I)
    W = incidence(ref, 14)
    m = mme(dyd[ref], ones(8, 1), [RandomEffect(W, Gi, σ²a)]; σ²e)
    b, (a,) = solutions(m, solve_mme(m))
    a = vec(a)
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

    # 3. Example 11.4: Back-solve SNP effects from GBLUP solutions
    g = snp_from_gblup(Z, Gi, a; k)
    check(
        "SNP effects back-solved from GBLUP",
        g,
        [0.087, -0.311, 0.262, -0.080, 0.110, 0.139, 0.000, 0.001, -0.061, -0.016],
    )

    # 4. Example 11.5: Selection index form of GBLUP
    dgv, rel = gblup_index(G, ref, dyd[ref] .- 9.944, σ²e / σ²a)
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
    r = cor(a[cand], dyd[cand])
    check("realized accuracy, reliability", [r, r^2], [0.49, 0.24]; atol = 6e-3)
    show_table(bull = 13:26, DGV = a, theoretical_rel = rel)

    (; G, a, g, dgv, rel)
end
