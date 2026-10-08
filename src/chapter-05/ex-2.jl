"""
    ex_05_2() -> NamedTuple

Example 5.2: Animal model with common environmental (full-sib / litter) effects
for piglet weaning weight (Mrode & Pocrnic, 2023, Section 5.3).

# Model Formulation
```math
y = Xb + Za + Wc + e
```
where:
- ``y`` is the vector of weaning weights for 10 piglets (animals 6 to 15).
- ``b`` is the vector of fixed sex effects (Male, Female).
- ``a`` is the vector of random animal additive genetic effects (breeding values)
  for all 15 animals in the pedigree, with ``\\mathrm{var}(a) = A \\sigma_a^2``.
- ``c`` is the vector of random common environmental effects (litters of dams 2, 4, and 5),
  with ``\\mathrm{var}(c) = I \\sigma_c^2``.
- ``e`` is the random residual error, with ``\\mathrm{var}(e) = I \\sigma_e^2``.
-  Variance components: ``\\sigma_a^2 = 20.0``, ``\\sigma_c^2 = 15.0``,
  ``\\sigma_e^2 = 65.0``. Total phenotypic variance ``\\sigma_P^2 = 100.0``. Heritability
  ``h^2 = 0.20``; common environment ratio ``c^2 = \\sigma_c^2 / \\sigma_P^2 = 0.15``.

# Key Concepts Illustrated
1.  **Litter / Full-Sib Effects**: Distinguishing genetic resemblance from shared
   maternal/environmental effects.
2.  **MME with Multiple Random Effects**: Adding an uncorrelated random factor with identity
   covariance.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed sex effects solutions.
- `a`: Vector of estimated breeding values for all 15 animals.
- `c`: Vector of common environmental (litter) effects for litters of dams 2, 4, and 5.
"""
function ex_05_2()
    println("Example 5.2: common environmental effects, piglet weaning weight")

    # 1. Pedigree and piglet weaning weight records
    ped = DataFrame(
        sire = [0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1],
        dam = [0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5],
    )
    piglet = 6:15
    sex = ["M", "F", "F", "F", "M", "F", "F", "M", "F", "M"]
    ww = [90.0, 70.0, 65.0, 98.0, 106.0, 60.0, 80.0, 100.0, 85.0, 68.0]

    # 2. Variance components
    σ²a = 20.0                                  # additive genetic variance
    σ²c = 15.0                                  # common environmental (litter) variance
    σ²e = 65.0                                  # residual variance

    # 3. Incidence matrices and MME setup
    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(piglet, nrow(ped))            # 15 animals in pedigree
    W, dams = incidence(ped.dam[piglet])        # litters of dams 2, 4, 5
    m = mme(
        ww,
        X,
        [
            RandomEffect(Z, ainv(ped), σ²a; name = "animal"),
            RandomEffect(W, I, σ²c; name = "litter"),
        ];
        σ²e,
    )

    # 4. Solve MME
    x = solve_mme(m)
    b, (a, c) = solutions(m, x)

    show_table(
        effect = ["male"; "female"; string.(1:15); "litter of " .* string.(dams)],
        solution = x,
    )
    check("sex", b, [91.493, 75.764])
    check(
        "animals",
        a,
        [
            -1.441,
            -1.175,
            1.441,
            1.441,
            -0.266,
            -1.098,
            -1.667,
            -2.334,
            3.925,
            2.895,
            -1.141,
            1.525,
            0.448,
            0.545,
            -3.819,
        ],
    )
    check("common environment", c, [-1.762, 2.161, -0.399])

    (; b, a = vec(a), c = vec(c))
end
