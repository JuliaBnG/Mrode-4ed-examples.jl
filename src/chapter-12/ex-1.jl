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

# Returns
A `NamedTuple` with fields:
- `Gt`: Tuned genomic relationship matrix.
- `Hi`: Single-step inverse relationship matrix ``H^{-1}``.
- `x`: Vector of ssGBLUP solutions for the mean and all 26 animals.
"""
function ex_12_1()
    println("Example 12.1: single-step GBLUP")

    # 1. Load pedigree, genotypes, and phenotypes
    (; ped, M, dyd, id, ref, σ²a, σ²e) = data_11()

    # 2. Construct raw G and pedigree relationship matrix A₂₂ for genotyped bulls
    _, p, _ = center_genotypes(M)
    G = grm(Matrix{Int8}(M'), p)
    A22 = nrm(ped, id)                            # Colleau's algorithm

    # 3. Blending and tuning G to A₂₂ (Chen et al., 2011)
    Gb = 0.95G + 0.05A22                          # blending with A₂₂
    Gt, a, b = tune_grm(Gb, A22)                  # tuning: Gt = a + b Gb
    check("tuning: Gt = a + b Gb", [a, b], [0.202, 0.709])

    # 4. Form H⁻¹: H⁻¹ = A⁻¹ + [0 0; 0 Gt⁻¹ - A₂₂⁻¹]
    Hi = hinv(ped, Gt, id)

    # 5. Setup and solve ssGBLUP MME
    W = incidence(id[ref], nrow(ped))
    m = mme(dyd[ref], ones(8, 1), [RandomEffect(W, Hi, σ²a)]; σ²e)
    x = solve_mme(m)

    show_table(animal = 1:26, ssGBLUP = x[2:end])
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

    (; Gt, Hi, x)
end
