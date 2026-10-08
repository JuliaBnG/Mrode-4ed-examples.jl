"""
    ex_13_1() -> NamedTuple

Example 13.1: Animal model with separate additive and dominance effects from pedigree
(Mrode & Pocrnic, 2023, Sections 13.2–13.3; Cockerham, 1954; Henderson, 1985).

# Model Formulation
Phenotypes are partitioned into additive genetic values and dominance deviations:
```math
y = Xb + Za + Zd + e
```
where:
- ``b`` is the fixed pen effect (pens 1 and 2).
-  ``a`` is the random additive genetic effect (breeding value), with
  ``\\mathrm{var}(a) = A \\sigma_a^2``.
-  ``d`` is the random dominance genetic deviation, with
  ``\\mathrm{var}(d) = D \\sigma_d^2``.
-  Pedigree dominance matrix ``D`` is constructed via Cockerham's rule (`drm(ped)`): the
  dominance relationship between individuals ``i`` and ``j`` with parents ``(s_i, d_i)`` and
  ``(s_j, d_j)`` is:
  ``D_{ij} = \\frac{1}{4}(A_{s_i s_j} A_{d_i d_j} + A_{s_i d_j} A_{d_i s_j})``.
-  Variance components: ``\\sigma_a^2 = 90.0``, ``\\sigma_d^2 = 80.0``,
  ``\\sigma_e^2 = 120.0``.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed pen effect solutions.
- `a`: Vector of estimated breeding values for all 12 pigs.
- `d`: Vector of estimated dominance deviations for all 12 pigs.
"""
function ex_13_1()
    println("Example 13.1: additive and dominance effects solved separately")

    # 1. Load data and setup variables
    (; ped, pig, pen, ww, σ²a, σ²d, σ²e) = data_13_1()
    N = nrow(ped)

    # 2. Construct pedigree dominance relationship matrix D and its inverse
    D = drm(ped)                                # Cockerham's rule
    Di = inv(D)
    check(
        "D, rows 7 and 11 (columns 7–12)",
        D[[7, 11], 7:12],
        [1 0 0.0625 0.0625 0.125 0.125; 0.125 0 0.125 0.125 1 0.25],
    )
    check(
        "D⁻¹, row 12",
        Di[12, 7:12],
        [-0.096, 0.000, -0.080, -0.080, -0.241, 1.0921];
        atol = 1e-3,
    )

    # 3. Setup incidence matrices and MME with additive and dominance effects
    X, _ = incidence(pen)
    Z = incidence(pig, N)
    m = mme(ww, X, [RandomEffect(Z, ainv(ped), σ²a), RandomEffect(Z, Di, σ²d)]; σ²e)

    # 4. Solve MME
    b, (a, d) = solutions(m, solve_mme(m))

    show_table(animal = 1:N, BV = vec(a), DV = vec(d))
    check("pen", b, [16.980, 20.030])
    check(
        "breeding values",
        a,
        [
            -0.160,
            -0.160,
            0.059,
            0.819,
            -0.320,
            1.259,
            0.555,
            -0.998,
            -0.350,
            -1.350,
            1.061,
            -0.039,
        ],
    )
    check(
        "dominance deviations",
        d,
        [0, 0, 0, 0, 0.136, 0.705, 0.237, -0.993, 0, -1.333, 1.428, -0.038],
    )

    (; b, a = vec(a), d = vec(d))
end
