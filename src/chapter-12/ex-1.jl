"""
    ex_12_1() -> NamedTuple

Example 12.1: Single-step GBLUP (ssGBLUP)
(Mrode & Pocrnic, 2023, Chapter 12; Aguilar et al., 2010; Christensen & Lund, 2010).

# Model Formulation
Single-step GBLUP integrates genotyped animals and ungenotyped relatives into a single
evaluation by replacing the pedigree relationship matrix ``A`` with the joint
pedigree–genomic relationship matrix ``H``:
```math
y = Xb + W a + e
```
where ``\\mathrm{var}(a) = H \\sigma_a^2``.
The inverse ``H^{-1}`` is computed directly without inverting the full ``H`` matrix:
```math
H^{-1} = A^{-1} + \\begin{bmatrix} 0 & 0 \\\\ 0 & G_t^{-1} - A_{22}^{-1} \\end{bmatrix}
```
where:
-  ``A_{22}`` is the pedigree relationship matrix for genotyped animals (computed using
  Colleau's algorithm).
- Blending: ``G_b = 0.95 G + 0.05 A_{22}`` to guarantee non-singularity.
-  Tuning (Chen et al., 2011): ``G_t = a + b G_b`` matches the mean diagonal and
  off-diagonal elements of ``G_b`` to ``A_{22}``, ensuring compatibility with the base
  population of ``A``.
- ``hinv(ped, Gt, id)`` builds ``H^{-1}`` directly.

# Key Concepts Illustrated
1. **Tuning, ``H^{-1}`` and the MME written out**: ``a, b`` from the 2 × 2 system
   matching mean diagonal and off-diagonal elements (compared with `tune_grm`);
   ``H^{-1}`` as ``A^{-1}`` plus ``G_t^{-1} - A_{22}^{-1}`` in the genotyped block
   (compared with `hinv`); ``[X'X\\ X'W;\\ W'X\\ W'W + \\lambda H^{-1}]`` (compared
   with `mme`).
2. **Tables**: ssGBLUP and pedigree-BLUP solutions of all 26 animals as columns
   of the pedigree. The candidates 22–26 are paternal half-sibs (sire 14) with
   ungenotyped dams without information: pedigree BLUP gives them all the same
   value (0.030), ssGBLUP separates them by their genotypes.

# Returns
A `NamedTuple` with fields:
- `Gt`: Tuned genomic relationship matrix.
- `Hi`: Single-step inverse relationship matrix ``H^{-1}``.
- `x`: Vector of ssGBLUP solutions for the mean and all 26 animals.
- `ped`: The pedigree with `genotyped`, `ssgblup` and `pblup` added.
"""
function ex_12_1()
    println("Example 12.1: single-step GBLUP")

    # 1. Pedigree (one row per animal), genotyped bulls (in the row order of M)
    (; ped, bulls, M, σ²a, σ²e) = data_11()
    id = bulls.id
    isref = bulls.set .== "reference"
    λ = σ²e / σ²a

    # 2. Construct raw G and pedigree relationship matrix A₂₂ for genotyped bulls
    _, p, _ = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)
    A22 = nrm(ped, id)                            # Colleau's algorithm

    # 3. Blending and tuning G to A₂₂ (Chen et al., 2011)
    Gb = 0.95G + 0.05A22                          # blending with A₂₂
    Gt, a, b = tune_grm(Gb, A22)                  # tuning: Gt = a + b Gb
    check("tuning: Gt = a + b Gb", [a, b], [0.202, 0.709])
    #    a and b match the mean diagonal and mean off-diagonal of Gb to A₂₂:
    #    mean diag(A₂₂) = a + b mean diag(Gb), mean offdiag(A₂₂) = a + b mean offdiag(Gb)
    off = .!I(length(id))
    md(K) = mean(diag(K))
    mo(K) = mean(K[off])
    ab = [1 md(Gb); 1 mo(Gb)] \ [md(A22), mo(A22)]
    check("a, b from the 2 × 2 system = tune_grm()", ab, [a, b]; atol = 1e-12)
    check("Gt = a + b Gb", Gt, a .+ b * Gb; atol = 1e-12)

    # 4. Form H⁻¹: H⁻¹ = A⁻¹ + [0 0; 0 Gt⁻¹ - A₂₂⁻¹]
    #    written out: start from A⁻¹ and add Gt⁻¹ - A₂₂⁻¹ to the block of the
    #    genotyped animals
    Hi = hinv(ped, Gt, id)
    Ai = ainv(ped)
    He = Matrix(Ai)
    He[id, id] += inv(Gt) - inv(A22)
    check("H⁻¹ = A⁻¹ + [0 0; 0 Gt⁻¹ - A₂₂⁻¹] (hinv)", Matrix(Hi), He; atol = 1e-10)

    # 5. ssGBLUP MME for the reference bulls, [X'X X'W; W'X W'W + λH⁻¹]
    y = bulls.dyd[isref]
    X = ones(count(isref), 1)
    W = incidence(id[isref], nrow(ped))
    lhs = [
        X'X  X'W
        W'X  W'W + λ * Hi
    ]
    m = mme(y, X, [RandomEffect(W, Hi, σ²a)]; σ²e)
    check("mme() LHS = [X'X X'W; W'X W'W + λH⁻¹]", m.lhs, lhs; atol = 1e-10)
    x = lhs \ [X'y; W'y]

    # 6. For comparison: the same model with A⁻¹ only (pedigree BLUP)
    xa = [X'X X'W; W'X W'W + λ * Ai] \ [X'y; W'y]
    ped.genotyped = in(Set(id)).(ped.id)
    ped.ssgblup = x[2:end]
    ped.pblup = xa[2:end]
    show_table(ped)
    check(
        "ssGBLUP, ungenotyped 1–12",
        x[2:13],
        [
            0.009,
            0.121,
            0.009,
            0.010,
            -0.250,
            -0.196,
            -0.001,
            0.026,
            -0.076,
            0.040,
            -0.137,
            0.136,
        ],
    )
    check(
        "ssGBLUP, genotyped 13–26",
        x[14:27],
        [
            0.048,
            0.082,
            0.039,
            0.202,
            -0.356,
            -0.254,
            0.098,
            -0.169,
            0.018,
            0.079,
            -0.164,
            0.101,
            0.039,
            0.245,
        ],
    )

    (; Gt, Hi, x, ped)
end
