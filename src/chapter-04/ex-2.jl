"""
    ex_04_2() -> NamedTuple

Example 4.2: Sire model for pre-weaning gain (WWG) of beef calves
(Mrode & Pocrnic, 2023, Section 4.4).

# Model Formulation
```math
y = Xb + Z_s s + e
```
where:
- ``y`` is the vector of pre-weaning gain records for 5 calves.
- ``b`` is the vector of fixed sex effects (Male, Female).
- ``s`` is the vector of random sire transmitting abilities (sires 1, 3, and 4),
  with ``\\mathrm{var}(s) = A_s \\sigma_s^2``.
- In a sire model, sire genetic variance is one quarter of additive genetic variance:
  ``\\sigma_s^2 = \\frac{1}{4} \\sigma_a^2 = 5.0``.
- Residual variance includes the remaining additive genetic variance and environmental
  variance: ``\\sigma_e^2 = \\sigma_P^2 - \\sigma_s^2 = 60.0 - 5.0 = 55.0``.
- Variance ratio: ``\\alpha_s = \\sigma_e^2 / \\sigma_s^2 = 55.0 / 5.0 = 11.0``.

# Key Concepts Illustrated
1. **Sire Pedigree and Inverse Relationship Matrix**:
   Sire 4 is a son of sire 1, so ``A_s^{-1}`` accounts for relationships among sires.
2.  **Comparison with Animal Model**: Sire solutions estimate transmitting abilities (half
   breeding value: ``\\hat{s} \\approx \\frac{1}{2} \\hat{a}``).

# Returns
A `NamedTuple` with fields:
- `b`: Fixed sex effect solutions.
- `s`: Vector of sire transmitting abilities.
- `x`: Full MME solution vector.
"""
function ex_04_2()
    println("Example 4.2: sire model")

    # 1. Phenotypic data and sire identification
    # records: sex of progeny, sire, WWG
    sex = ["M", "F", "F", "M", "M"]
    sire = [1, 3, 1, 4, 3]
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]

    # 2. Sire pedigree (sires recoded: 1 → 1, 3 → 2, 4 → 3; sire 4 is a son of 1)
    sires = [1, 3, 4]
    sped = DataFrame(sire = [0, 0, 1], dam = [0, 0, 0])

    # 3. Variance components
    σ²s = 0.25 * 20.0                           # sire variance: (1/4) σ²a
    σ²e = 60.0 - σ²s                            # residual variance: σ²P - σ²s = 55.0

    # 4. Incidence matrices and MME setup
    X, _ = incidence(sex; levels = ["M", "F"])
    Z, _ = incidence(sire; levels = sires)
    m = mme(wwg, X, [RandomEffect(Z, ainv(sped), σ²s; name = "sire")]; σ²e)

    # 5. Solve MME and check solutions
    x = solve_mme(m)
    b, (s_hat,) = solutions(m, x)

    show_table(effect = ["male", "female", "sire " .* string.(sires)...], solution = x)
    check("A⁻¹ of sires", Matrix(ainv(sped)), [1.333 0 -0.667; 0 1 0; -0.667 0 1.333])
    check(
        "MME coefficient matrix",
        Matrix(m.lhs),
        [3 0 1 1 1; 0 2 1 1 0; 1 1 16.666 0 -7.334; 1 1 0 13 0; 1 0 -7.334 0 15.666],
    )
    check("solutions", x, [4.336, 3.382, 0.022, 0.014, -0.043])

    (; b, s = vec(s_hat), x)
end
