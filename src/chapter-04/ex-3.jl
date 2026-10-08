"""
    ex_04_3() -> NamedTuple

Example 4.3: Reduced Animal Model (RAM) for pre-weaning gain of beef calves
(Mrode & Pocrnic, 2023, Section 4.5; Quaas & Pollak, 1980).

# Model Formulation
The Reduced Animal Model partitions breeding values into parent breeding values
``a_p`` and non-parent Mendelian sampling deviations ``m_0``:
```math
y = Xb + W a_p + e^*
```
where:
- Only parents (animals 1 to 6) are directly evaluated in the MME.
- ``W = Z_p + \\frac{1}{2}(Z_{sp} + Z_{dp})`` maps records to parents.
- Non-parents (calves 7 and 8) have their equations absorbed into the parents'
  equations, reducing the dimension of the MME from 10 to 8.
-  Residual variance for non-parents is modified to include their Mendelian sampling
  variance: ``\\mathrm{var}(e^*_i) = \\sigma_e^2 + \\frac{1}{2} \\sigma_a^2`` (if both
  parents are known). The diagonal matrix ``R^{-1}`` (`rinv`) weights records accordingly.

# Back-solving
After solving for parent breeding values ``\\hat{a}_p`` and fixed effects ``\\hat{b}``,
breeding values of non-parents are back-solved:
```math
\\hat{a}_i = \\mathrm{PA}_i + k_i \\left( y_i - x_i' \\hat{b} - \\mathrm{PA}_i \\right),
\\quad \\mathrm{PA}_i = \\tfrac{1}{2}(\\hat{a}_{s_i} + \\hat{a}_{d_i}),
\\quad k_i = \\frac{m_i \\sigma_a^2}{\\sigma_e^2 + m_i \\sigma_a^2}
```
(Eqn 4.26), where ``m_i`` is the Mendelian-sampling variance factor (``\\tfrac12`` for
non-inbred known parents, so ``k_i = 10/50 = 0.2`` here).

# Returns
A `NamedTuple` with fields:
- `b`: Fixed sex effects solutions.
- `a`: Full vector of EBVs for all 8 animals (parents + back-solved non-parents).
- `ap`: Parent breeding value solutions from RAM MME.
- `rinv`: Diagonal elements of ``R^{-1}``.
"""
function ex_04_3()
    println("Example 4.3: reduced animal model")

    # 1. Pedigree, records, and variance components
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = [4, 5, 6, 7, 8]
    sex = ["M", "F", "F", "M", "M"]
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]
    σ²a, σ²e = 20.0, 40.0

    # 2. RAM design matrices: W, R⁻¹, parents list, and A_p⁻¹
    W, rinv, parents, Ap⁻¹ = ram_design(ped, calf, σ²a, σ²e)   # Ap⁻¹: parents only

    # 3. Fixed effects incidence matrix and RAM MME setup
    X, _ = incidence(sex; levels = ["M", "F"])
    m = mme(wwg, X, [RandomEffect(W, Ap⁻¹, σ²a; name = "parents")]; Rinv = rinv)

    # 4. Solve RAM MME
    x = solve_mme(m)
    b, (ap,) = solutions(m, x)

    check("R⁻¹", rinv, [0.025, 0.025, 0.025, 0.020, 0.020])
    check("W'R⁻¹y", (W' * (rinv .* wwg)), [0, 0, 0.050, 0.148, 0.107, 0.148])
    check(
        "sex and parent solutions",
        x,
        [4.358, 3.404, 0.098, -0.019, -0.041, -0.009, -0.186, 0.177],
    )
    check("non-zeros in the coefficient matrix", nnz(m.lhs), 38)

    # 5. Back-solve breeding values of non-parents (calves 7 and 8)
    a = ram_backsolve(ped, calf, wwg - X * b, vec(ap), parents, σ²a, σ²e)

    show_table(animal = 1:8, ebv = a)
    check("back-solved non-parents 7 and 8", a[7:8], [-0.249, 0.183])

    (; b, a, ap = vec(ap), rinv)
end
