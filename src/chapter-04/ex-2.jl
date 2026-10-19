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
1. **Same data as Example 4.1**: one `DataFrame` with pedigree and records
   of all 8 animals. The sires with progeny records are taken from it.
2. **Sire Pedigree and Inverse Relationship Matrix**: the sire pedigree is
   derived from `ped`, in original IDs. Only the sires of sires are kept
   (sire 4 is a son of sire 1); dams are set to unknown, as in the book.
   For `ainv` the sires are recoded `1:3` in order of their IDs, so that
   parents still precede progeny.
3. **MME**, written out explicitly for ``R = I\\sigma_e^2``:
   ```math
   \\begin{bmatrix}
   X'X & X'Z_s \\\\
   Z_s'X & Z_s'Z_s + \\alpha_s A_s^{-1}
   \\end{bmatrix}
   \\begin{bmatrix}\\hat{b} \\\\ \\hat{s}\\end{bmatrix} =
   \\begin{bmatrix}X'y \\\\ Z_s'y\\end{bmatrix},
   ```
   and compared with `mme`. See [`ex_04_1`](@ref) for the general
   ``R^{-1}`` form.
4. **Comparison with the animal model**: the transmitting ability is half the
   breeding value, ``s = \\frac{1}{2} a``, but the *estimates* need not agree:
   the sire model uses only progeny records, and ignores the sires' own records
   (sire 4), the dams and their relationships. Here even the ranking of the
   three sires differs from ``\\frac{1}{2}\\hat{a}`` of Example 4.1.

# Returns
A `NamedTuple` with fields:
- `sires`: Table of sires with the sire solutions `s` and ``\\frac{1}{2}\\hat{a}``
  of Example 4.1.
- `b`: Fixed sex effect solutions.
- `s`: Vector of sire transmitting abilities.
- `x`: Full MME solution vector.
"""
function ex_04_2()
    println("Example 4.2: sire model")

    # 1. Pedigree and phenotypic data, as in Example 4.1
    ped = DataFrame(
        calf = 1:8,
        sex = [missing, missing, missing, "M", "F", "F", "M", "M"],
        sire = [0, 0, 0, 1, 3, 1, 4, 3],
        dam = [0, 0, 0, 0, 2, 2, 5, 6],
        wwg = [missing, missing, missing, 4.5, 2.9, 3.9, 3.5, 5.0],
    )
    rec = dropmissing(ped, :wwg)                # the animals with records

    # 2. Sire pedigree, in original IDs: sires with progeny records; a sire's
    #    sire is kept if it is in the list (sire 4 is a son of 1); dams unknown.
    sires = DataFrame(id = sort(unique(rec.sire)))
    sires.sire = [ped.sire[id] in sires.id ? ped.sire[id] : 0 for id in sires.id]
    sires.dam .= 0
    println(sires)

    # 3. Variance components
    σ²s = 0.25 * 20.0                           # sire variance: (1/4) σ²a
    σ²e = 60.0 - σ²s                            # residual variance: σ²P - σ²s = 55.0
    α = σ²e / σ²s                               # α = 11.0

    # 4. Incidence matrices, A_s⁻¹ and the MME
    y = rec.wwg
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Z, _ = incidence(rec.sire; levels = sires.id)
    # ainv needs IDs 1:n: recode sires in ID order (1 → 1, 3 → 2, 4 → 3),
    # so that parents still precede progeny
    code = Dict(id => i for (i, id) in enumerate(sires.id))
    Asi = ainv(DataFrame(sire = [get(code, p, 0) for p in sires.sire], dam = sires.dam))
    #! format: off
    lhs = [
        X'X  X'Z
        Z'X  Z'Z + α * Asi
    ]
    #! format: on
    rhs = [X'y; Z'y]
    x = lhs \ rhs

    println("\nCoefficient matrix [X'X X'Z; Z'X Z'Z + αA_s⁻¹]:")
    show(stdout, "text/plain", round.(Matrix(lhs); digits = 3))
    println()
    println("\nRight-hand side [X'y; Z'y]: ", round.(rhs; digits = 3))
    check("A⁻¹ of sires", Matrix(Asi), [1.333 0 -0.667; 0 1 0; -0.667 0 1.333])
    check(
        "MME coefficient matrix",
        Matrix(lhs),
        [3 0 1 1 1; 0 2 1 1 0; 1 1 16.666 0 -7.334; 1 1 0 13 0; 1 0 -7.334 0 15.666],
    )

    m = mme(y, X, [RandomEffect(Z, Asi, σ²s; name = "sire")]; σ²e)
    check("mme() LHS = [X'X X'Z; Z'X Z'Z+αA_s⁻¹]", m.lhs, lhs; atol = 1e-12)

    # 5. Solutions
    b = x[m.blocks[1]]
    s = x[m.blocks[2]]
    sires.s = s
    println("\nSex effects:")
    show_table(sex = ["M", "F"], solution = b)
    check("solutions", x, [4.336, 3.382, 0.022, 0.014, -0.043])

    # 6. Half the breeding values of the sires from the animal model of
    #    Example 4.1 (same data, σ²a = 20, σ²e = 40), for comparison only
    Z_a = incidence(rec.calf, nrow(ped))
    m_a = mme(y, X, [RandomEffect(Z_a, ainv(ped), 20.0; name = "animal")]; σ²e = 40.0)
    a = solve_mme(m_a)[m_a.blocks[2]]
    sires.half_ebv = a[sires.id] ./ 2
    println("\nSires: sire-model solutions and ½ EBV of the animal model (Example 4.1):")
    show_table(sires)

    (; sires, b, s, x)
end
