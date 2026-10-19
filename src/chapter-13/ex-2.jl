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

# Key Concepts Illustrated
1. **One genetic effect**: the MME ``[X'X\\ X'Z;\\ Z'X\\ Z'Z + \\sigma_e^2 V^{-1}]`` are written
   out and compared with `mme`; ``\\hat{a}`` and ``\\hat{d}`` are recovered from ``\\hat{g}``
   and compared with the separate model of Example 13.1, fitted here.
2. **Tables**: ``\\hat{g}``, ``\\hat{a}`` and ``\\hat{d}`` are added as columns of the pig table.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed pen effects.
- `g`: Vector of total genetic merit ``\\hat{g} = \\hat{a} + \\hat{d}`` for all 12 pigs.
- `a`: Recovered additive breeding values.
- `d`: Recovered dominance deviations.
- `ped`: The pigs with columns `g`, `bv` and `dv` added.
"""
function ex_13_2()
    println("Example 13.2: total genetic merit")

    # 1. Pigs (one row per pig, records on pigs 5–12) and the covariance matrix
    #    of total genetic merit, V = Aσ²a + Dσ²d
    (; ped, σ²a, σ²d, σ²e) = data_13_1()
    rec = dropmissing(ped, :ww)
    A, D = nrm(ped), drm(ped)
    V = σ²a * A + σ²d * D
    Vi = inv(V)

    # 2. MME for g with one genetic effect (R = Iσ²e): [X'X X'Z; Z'X Z'Z + σ²e V⁻¹]
    y = rec.ww
    X, _ = incidence(rec.pen)
    Z = incidence(rec.pig, nrow(ped))
    lhs = [
        X'X  X'Z
        Z'X  Z'Z + σ²e * Vi
    ]
    m = mme(y, X, [RandomEffect(Z, Vi, 1.0)]; σ²e)
    check("mme() LHS = [X'X X'Z; Z'X Z'Z + σ²e V⁻¹]", m.lhs, lhs; atol = 1e-10)
    x = lhs \ [X'y; Z'y]
    b, g = x[m.blocks[1]], x[m.blocks[2]]

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
    #    â = σ²a A V⁻¹ g,   d̂ = σ²d D V⁻¹ g, so that â + d̂ = ĝ
    a = σ²a * A * (Vi * g)
    d = σ²d * D * (Vi * g)
    check("â + d̂ = ĝ", a + d, g; atol = 1e-10)

    # 4. The separate model of Example 13.1, T = [X Z Z], gives the same â and d̂
    T = [X Z Z]
    Σ = blockdiag(spzeros(2, 2), sparse(σ²e / σ²a * ainv(ped)), sparse(σ²e / σ²d * inv(D)))
    xs = (T'T + Σ) \ (T'y)
    N = nrow(ped)
    check("â = σ²a A V⁻¹ ĝ (Example 13.1)", a, xs[2 .+ (1:N)]; atol = 1e-8)
    check("d̂ = σ²d D V⁻¹ ĝ (Example 13.1)", d, xs[2 + N .+ (1:N)]; atol = 1e-8)

    ped.g, ped.bv, ped.dv = g, a, d
    show_table(ped)

    (; b, g, a, d, ped)
end
