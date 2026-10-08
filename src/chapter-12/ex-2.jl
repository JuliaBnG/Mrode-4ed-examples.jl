"""
    ex_12_2() -> NamedTuple

Example 12.2: Algorithm for Proven and Young (APY) inverse of ``G``
(Mrode & Pocrnic, 2023, Section 12.4; Misztal et al., 2014; Misztal, 2016).

# Model Formulation and APY Theory
For large populations, inverting ``G`` directly is computationally prohibitive (cubic
complexity ``O(n^3)``). APY approximates ``G^{-1}`` with linear complexity by partitioning
genotyped animals into **core** animals (c) and **non-core** animals (n):
```math
G_{\\mathrm{APY}}^{-1} =
\\begin{bmatrix}
G_{cc}^{-1} + G_{cc}^{-1} G_{cn} M_{nn}^{-1} G_{nc} G_{cc}^{-1} & -G_{cc}^{-1} G_{cn}
    M_{nn}^{-1} \\\\
-M_{nn}^{-1} G_{nc} G_{cc}^{-1} & M_{nn}^{-1}
\\end{bmatrix}
```
where ``M_{nn}`` is a strictly diagonal matrix with elements:
```math
m_{ii} = g_{ii} - g_{ic} G_{cc}^{-1} g_{ci}
```
representing the conditional variance of non-core animal ``i`` given the core animals.

# Key Concepts Illustrated
1. **Core Animal Selection**: Bulls 15, 20, 22, and 26 chosen as core animals.
2.  **Sparsity of APY Inverse**: The non-core block of ``G_{\\mathrm{APY}}^{-1}`` is
   strictly diagonal.
3.  **Comparison of DGVs**: Evaluating correlation between DGVs computed using exact
   ``G^{-1}`` versus APY ``G_{\\mathrm{APY}}^{-1}`` (here ``r \\approx 0.95``).

# Note on Text Discrepancy
The book omits the minus sign on the DGV for bull 17 (prints 0.471 instead of -0.471).

# Returns
A `NamedTuple` with fields:
- `Gi`: APY inverse of the genomic relationship matrix ``G_{\\mathrm{APY}}^{-1}``.
- `a_apy`: Direct genomic values computed using APY inverse.
- `a_full`: Direct genomic values computed using exact full inverse.
"""
function ex_12_2()
    println("Example 12.2: APY inverse of G")

    # 1. Load data and construct full G
    (; M, dyd, id, ref, σ²a, σ²e) = data_11()
    _, p, _ = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)

    # 2. Define core bulls (15, 20, 22, 26) and non-core bulls
    core = findall(in([15, 20, 22, 26]), id)
    nc = setdiff(1:14, core)

    # 3. Compute APY inverse
    Gi = apy_ginv(G, core)
    Mnn = 1 ./ diag(Gi)[nc]
    check(
        "diag(Mₙₙ)",
        Mnn,
        [0.234, 0.204, 0.600, 0.848, 0.485, 0.420, 1.107, 0.104, 0.644, 0.102],
    )

    # Note: the book prints G⁻¹_APY with core bulls ordered first
    o = [core; nc]
    check(
        "G⁻¹_APY: first row (bull 15, APY order)",
        Matrix(Gi)[o[1], o],
        [
            4.425,
            0.373,
            -0.527,
            0.412,
            -0.795,
            2.731,
            -0.349,
            -0.658,
            0.959,
            0.776,
            0.483,
            -0.447,
            -0.108,
            1.828,
        ],
    )
    check("non-core block is diagonal", nnz(Gi[nc, nc]), length(nc); atol = 0)

    # 4. Compare GBLUP solutions with APY vs full inverse
    W = incidence(ref, 14)
    dgv(K) = solve_mme(mme(dyd[ref], ones(8, 1), [RandomEffect(W, K, σ²a)]; σ²e))[2:end]
    a_apy = dgv(Gi)
    a_full = dgv(inv(G + 0.01I))

    show_table(bull = id, full = a_full, APY = a_apy)
    # Note: the book omits the minus sign of bull 17 (-0.471)
    check(
        "DGV with G⁻¹_APY",
        a_apy,
        [
            0.065,
            0.119,
            0.030,
            0.365,
            -0.471,
            -0.244,
            -0.055,
            -0.220,
            0.010,
            0.127,
            -0.229,
            0.097,
            -0.007,
            0.363,
        ],
    )
    check("cor(full, APY)", cor(a_full, a_apy), 0.95; atol = 0.005)

    (; Gi, a_apy, a_full)
end
