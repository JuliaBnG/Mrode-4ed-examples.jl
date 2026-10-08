"""
    ex_13_5() -> NamedTuple

Example 13.5: Genomic model with additive, dominance, and additive × additive epistatic
effects (Mrode & Pocrnic, 2023, Section 13.5; Su et al., 2012; Vitezica et al., 2017).

# Model Formulation
Extends genomic evaluation to capture non-allelic gene interaction (epistasis):
```math
y = Xb + Z a_g + Z d_g + Z aa_g + e
```
where:
- ``a_g`` is the genomic additive effect (variance ``\\sigma_a^2 = 90.0``).
- ``d_g`` is the genomic dominance effect (variance ``\\sigma_d^2 = 80.0``).
- ``aa_g`` is the additive × additive epistatic effect (variance ``\\sigma_{aa}^2 = 6.0``).
-  The epistatic relationship matrix ``G_{AA}`` is obtained as the normalized Hadamard
  (Schur) product of the additive relationship matrix:
  ```math
  G_{AA} = \\frac{G \\odot G}{\\mathrm{tr}(G \\odot G) / n}
  ```
  built with `epistatic_grm(G)`.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed pen solutions.
- `a`: Additive genomic solutions for all 15 pigs.
- `d`: Dominance genomic solutions for all 15 pigs.
- `aa`: Additive × additive epistatic genomic solutions for all 15 pigs.
"""
function ex_13_5()
    println("Example 13.5: additive × additive epistasis")

    # 1. Load data and setup variance components
    (; M, pig, pen, ww, σ²a, σ²d, σ²e) = data_13_3()
    σ²aa = 6.0

    # 2. Construct relationship matrices: G, D, and epistatic GAA = G ⊙ G (normalized)
    p = vec(mean(M; dims = 1)) ./ 2
    G = grm(Matrix{Int8}(M'), p)
    D = grm(Matrix{Int8}(M'), p; method = :dominance)
    GAA = epistatic_grm(G)

    check(
        "G_AA, last row",
        GAA[15, :],
        [
            0.136,
            0.012,
            0.422,
            0.511,
            0.224,
            0.404,
            0.029,
            0.237,
            0.045,
            0.295,
            0.224,
            0.029,
            0.324,
            0.014,
            1.379,
        ],
    )

    # 3. Setup MME with additive, dominance, and epistatic random effects
    X, _ = incidence(pen)
    Z = incidence(pig, 15)
    m = mme(
        ww,
        X,
        [
            RandomEffect(Z, inv(G + 0.01I), σ²a),
            RandomEffect(Z, inv(D + 0.01I), σ²d),
            RandomEffect(Z, inv(GAA + 0.01I), σ²aa),
        ];
        σ²e,
    )

    # 4. Solve MME
    b, (a, d, aa) = solutions(m, solve_mme(m))

    show_table(animal = 1:15, additive = vec(a), dominance = vec(d), epistatic = vec(aa))
    check("pen", b, [17.453, 20.833])
    check(
        "additive",
        a,
        [
            -0.108,
            0.566,
            -0.916,
            0.248,
            0.020,
            1.545,
            0.732,
            -0.934,
            -1.288,
            -2.045,
            1.352,
            0.718,
            0.387,
            -0.862,
            0.585,
        ],
    )
    check(
        "dominance",
        d,
        [
            0.191,
            0.920,
            -0.975,
            0.506,
            0.100,
            0.351,
            -0.494,
            -0.836,
            -0.710,
            -0.343,
            1.085,
            -0.506,
            1.768,
            0.496,
            1.094,
        ],
    )
    check(
        "epistatic",
        aa,
        [
            0.035,
            -0.026,
            -0.029,
            0.082,
            -0.029,
            0.044,
            -0.056,
            0.002,
            -0.090,
            -0.153,
            0.065,
            -0.057,
            0.078,
            -0.029,
            0.126,
        ],
    )

    (; b, a = vec(a), d = vec(d), aa = vec(aa))
end
