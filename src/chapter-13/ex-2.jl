"""
    ex_13_2() -> NamedTuple

Example 13.2: Solving directly for total genetic merit ``g = a + d``
(Mrode & Pocrnic, 2023, Section 13.3.2).

# Model Formulation
Instead of setting up two sets of genetic equations (additive ``a`` and dominance ``d``),
total genetic merit ``g = a + d`` can be evaluated directly in a single random effect:
```math
y = Xb + Zg + e
```
where:
- ``\\mathrm{var}(g) = V = A \\sigma_a^2 + D \\sigma_d^2``.
- Inverse covariance: ``V^{-1} = (A \\sigma_a^2 + D \\sigma_d^2)^{-1}``.
-  Post-recovery of individual additive and dominance components from total merit
  ``\\hat{g}``:
  ```math
  \\hat{a} = \\sigma_a^2 A V^{-1} \\hat{g}, \\quad \\hat{d} = \\sigma_d^2 D V^{-1} \\hat{g}
  ```
- This yields identical solutions to the separate model of Example 13.1 while halving the
  number of random genetic equations.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed pen effects.
- `g`: Vector of total genetic merit ``\\hat{g} = \\hat{a} + \\hat{d}`` for all 12 pigs.
- `a`: Recovered additive breeding values.
- `d`: Recovered dominance deviations.
"""
function ex_13_2()
    println("Example 13.2: total genetic merit")

    # 1. Load data and setup covariance matrices
    (; ped, pig, pen, ww, σ²a, σ²d, σ²e) = data_13_1()
    A, D = nrm(ped), drm(ped)
    V = σ²a * A + σ²d * D
    Vi = inv(V)

    # 2. Setup and solve MME for total genetic merit g
    X, _ = incidence(pen)
    m = mme(ww, X, [RandomEffect(incidence(pig, nrow(ped)), Vi, 1.0)]; σ²e)
    b, (g,) = solutions(m, solve_mme(m))
    g = vec(g)

    check("pen", b, [16.980, 20.030])
    check(
        "total genetic merit",
        g,
        [
            -0.160,
            -0.160,
            0.059,
            0.819,
            -0.184,
            1.963,
            0.792,
            -1.991,
            -0.349,
            -2.683,
            2.489,
            -0.078,
        ],
    )

    # 3. Post-recovery of additive and dominance components:
    #    â = σ²a A V⁻¹ g,   d̂ = σ²d D V⁻¹ g
    a = σ²a * A * (Vi * g)
    d = σ²d * D * (Vi * g)

    # 4. Verify equivalence with separate solutions from Example 13.1
    r = ex_13_1()
    check("â = σ²a A M⁻¹ g (Example 13.1)", a, r.a; atol = 1e-8)
    check("d̂ = σ²d D M⁻¹ g (Example 13.1)", d, r.d; atol = 1e-8)

    (; b, g, a, d)
end
