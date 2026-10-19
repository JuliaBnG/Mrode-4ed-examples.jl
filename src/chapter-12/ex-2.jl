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
3. **The APY inverse written out** from its blocks and compared with `apy_ginv`;
   GBLUP MME written out and compared with `mme`.
4.  **Comparison of DGVs**: Evaluating correlation between DGVs computed using exact
   ``G^{-1}`` versus APY ``G_{\\mathrm{APY}}^{-1}`` (here ``r \\approx 0.95``).

# Note on Text Discrepancy
The book omits the minus sign on the DGV for bull 17 (prints 0.471 instead of -0.471).

# Returns
A `NamedTuple` with fields:
- `Gi`: APY inverse of the genomic relationship matrix ``G_{\\mathrm{APY}}^{-1}``.
- `a_apy`: Direct genomic values computed using APY inverse.
- `a_full`: Direct genomic values computed using exact full inverse.
- `bulls`: The bulls with `core`, `dgv_full` and `dgv_apy` added.
"""
function ex_12_2()
    println("Example 12.2: APY inverse of G")

    # 1. Genotyped bulls (in the row order of M) and the full G
    (; bulls, M, σ²a, σ²e) = data_11()
    id = bulls.id
    isref = bulls.set .== "reference"
    _, p, _ = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)

    # 2. Define core bulls (15, 20, 22, 26) and non-core bulls
    core = findall(in([15, 20, 22, 26]), id)
    nc = setdiff(1:14, core)

    # 3. APY inverse, written out: with P = G_cc⁻¹ G_cn and the diagonal
    #    M_nn = diag(g_ii - g_ic G_cc⁻¹ g_ci),
    #    G⁻¹_APY = [G_cc⁻¹ + P M_nn⁻¹ P'  -P M_nn⁻¹; -M_nn⁻¹ P'  M_nn⁻¹]
    Gi = apy_ginv(G, core)
    Gcci = inv(G[core, core])
    Pc = Gcci * G[core, nc]
    Mnn = [G[i, i] - G[i, core]' * Gcci * G[core, i] for i in nc]
    Ge = zeros(size(G))
    Ge[core, core] = Gcci + Pc * Diagonal(1 ./ Mnn) * Pc'
    Ge[core, nc] = -Pc * Diagonal(1 ./ Mnn)
    Ge[nc, core] = Ge[core, nc]'
    Ge[nc, nc] = Diagonal(1 ./ Mnn)
    check("G⁻¹_APY from blocks = apy_ginv()", Ge, Matrix(Gi); atol = 1e-10)
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

    # 4. GBLUP with G⁻¹_APY and with the full inverse: [X'X X'W; W'X W'W + λK⁻¹]
    y = bulls.dyd[isref]
    X = ones(count(isref), 1)
    W = incidence(findall(isref), nrow(bulls))
    λ = σ²e / σ²a
    function dgv(lab, Ki)
        lhs = [X'X X'W; W'X W'W + λ * Ki]
        m = mme(y, X, [RandomEffect(W, Ki, σ²a)]; σ²e)
        check("$lab: mme() LHS = [X'X X'W; W'X W'W + λK⁻¹]", m.lhs, lhs; atol = 1e-10)
        (lhs\[X'y; W'y])[2:end]
    end
    a_apy = dgv("APY", Gi)
    a_full = dgv("full", inv(G + 0.01I))
    bulls.core = in(Set([15, 20, 22, 26])).(id)
    bulls.dgv_full, bulls.dgv_apy = a_full, a_apy
    show_table(bulls)
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

    (; Gi, a_apy, a_full, bulls)
end
