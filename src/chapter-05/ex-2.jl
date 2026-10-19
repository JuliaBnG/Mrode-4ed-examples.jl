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
1. **One table for the data**: pedigree, sex, litter and weaning weight of
   all 15 animals in one `DataFrame`; the parents (1–5) have no records.
   Each dam has one litter here, so litters are named by their dam, but a
   litter is a level of its own factor: a dam with several litters would
   have several. The litter solutions are kept in a table of litters and
   joined back onto the piglets.
2. **Litter / Full-Sib Effects**: Distinguishing genetic resemblance from shared
   maternal/environmental effects.
3. **MME with Multiple Random Effects**: an uncorrelated random factor with identity
   covariance. With ``\\lambda_a = \\sigma_e^2/\\sigma_a^2`` and
   ``\\lambda_c = \\sigma_e^2/\\sigma_c^2``,
   ```math
   \\begin{bmatrix}
   X'X & X'Z & X'W \\\\
   Z'X & Z'Z + \\lambda_a A^{-1} & Z'W \\\\
   W'X & W'Z & W'W + \\lambda_c I
   \\end{bmatrix}
   \\begin{bmatrix}\\hat{b} \\\\ \\hat{a} \\\\ \\hat{c}\\end{bmatrix} =
   \\begin{bmatrix}X'y \\\\ Z'y \\\\ W'y\\end{bmatrix},
   ```
   written out explicitly and compared with `mme`.

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `ebv` and `c` (the effect of the piglet's litter) added.
- `litters`: Table of litters with their solutions `c`.
- `b`: Fixed sex effects solutions.
- `a`: Vector of estimated breeding values for all 15 animals.
- `c`: Vector of common environmental (litter) effects for litters of dams 2, 4, and 5.
"""
function ex_05_2()
    println("Example 5.2: common environmental effects, piglet weaning weight")

    # 1. Pedigree and piglet weaning weight records, one row per animal
    #    (IDs are row numbers; parents 1–5 have no records)
    no5 = fill(missing, 5)
    ped = DataFrame(
        id = 1:15,
        sire = [0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1],
        dam = [0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5],
        sex = [no5; ["M", "F", "F", "F", "M", "F", "F", "M", "F", "M"]],
        litter = [no5; [2, 2, 2, 4, 4, 4, 4, 5, 5, 5]],    # named by the dam
        ww = [no5; [90.0, 70.0, 65.0, 98.0, 106.0, 60.0, 80.0, 100.0, 85.0, 68.0]],
    )
    @assert ped.id == 1:nrow(ped)
    rec = dropmissing(ped, :ww)                 # the piglets with records
    println(ped)

    # 2. Variance components and variance ratios
    σ²a = 20.0                                  # additive genetic variance
    σ²c = 15.0                                  # common environmental (litter) variance
    σ²e = 65.0                                  # residual variance
    λa = σ²e / σ²a                              # 3.25
    λc = σ²e / σ²c                              # 4.333

    # 3. Incidence matrices
    y = rec.ww
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Z = incidence(rec.id, nrow(ped))            # 15 animals in pedigree
    W, lv = incidence(rec.litter)               # litters 2, 4, 5
    litters = DataFrame(litter = lv)
    Ai = ainv(ped)

    # 4. MME, written out (R = Iσ²e)
    lhs = [
        X'X  X'Z               X'W
        Z'X  Z'Z + λa * Ai     Z'W
        W'X  W'Z               W'W + λc * I
    ]
    rhs = [X'y; Z'y; W'y]
    x = lhs \ rhs

    m = mme(
        y,
        X,
        [
            RandomEffect(Z, Ai, σ²a; name = "animal"),
            RandomEffect(W, I, σ²c; name = "litter"),
        ];
        σ²e,
    )
    check("mme() LHS = explicit form", m.lhs, lhs; atol = 1e-12)

    # 5. Solutions
    b = x[m.blocks[1]]
    a = x[m.blocks[2]]
    c = x[m.blocks[3]]
    println("\nSex effects:")
    show_table(sex = ["M", "F"], solution = b)
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

    # 6. Results into the tables: litter solutions, joined back onto the piglets
    ped.ebv = a
    litters.c = c
    println("\nLitters:")
    show_table(litters)
    leftjoin!(ped, litters; on = :litter, matchmissing = :notequal)   # parents: no litter

    println("\nData with solutions:")
    show_table(ped)

    (; ped, litters, b, a, c)
end
