"""
    snpblup(Z, y, σ²a, σ²e, k; w = nothing) -> Vector{Float64}

Fit a SNP-BLUP (ridge regression BLUP) model solving for the overall mean and random marker
effects ``g``:
```math
y = \\mathbf{1}\\mu + Z g + e
```
where ``\\mathrm{var}(g) = I \\frac{\\sigma_a^2}{k}`` with scaling
``k = 2 \\sum p_j (1 - p_j)``. Optionally weights residual errors with vector `w` (e.g.
Effective Daughter Contributions).
"""
function snpblup(Z, y, σ²a, σ²e, k; w = nothing)
    m =
        w === nothing ? mme(y, ones(length(y), 1), [RandomEffect(Z, I, σ²a / k)]; σ²e) :
        mme(y, ones(length(y), 1), [RandomEffect(Z, I, σ²a / k)]; Rinv = w ./ σ²e)
    solve_mme(m)
end

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

# Returns
A `NamedTuple` with fields:
- `x`: Unweighted SNP-BLUP solutions (mean and 10 SNP effects).
- `xw`: EDC-weighted SNP-BLUP solutions.
"""
function ex_11_2()
    println("Example 11.2: SNP-BLUP")

    # 1. Load data and center genotypes
    (; M, dyd, edc, ref, cand, σ²a, σ²e) = data_11()
    Z, p, k = center_genotypes(M)
    check("2Σpq and α", [k, k * σ²e / σ²a], [3.5383, 24.598]; atol = 2e-3)

    # 2. Fit unweighted and EDC-weighted SNP-BLUP
    x = snpblup(Z[ref, :], dyd[ref], σ²a, σ²e, k)
    xw = snpblup(Z[ref, :], dyd[ref], σ²a, σ²e, k; w = edc[ref])

    show_table(effect = ["mean"; "SNP " .* string.(1:10)], unweighted = x, weighted = xw)
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

    # 3. Predict DGV for reference animals and candidates: DGV = Z ĝ
    check(
        "unweighted DGV (reference, candidates)",
        Z * x[2:end],
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
        Z * xw[2:end],
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

    (; x, xw)
end
