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

# Key Concepts Illustrated
1. **One table for the data**: pedigree, pen and weaning weight of all 12 pigs;
   breeding values and dominance deviations are added as columns.
2. **``D`` by Cockerham's rule**, written out and compared with `drm`.
3. **MME** as ``T'T + \\Sigma`` with ``T = [X\\ Z\\ Z]`` (the same ``Z`` for additive and
   dominance effects) and ``\\Sigma`` holding ``\\lambda_a A^{-1}`` and
   ``\\lambda_d D^{-1}``; compared with `mme`.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed pen effect solutions.
- `a`: Vector of estimated breeding values for all 12 pigs.
- `d`: Vector of estimated dominance deviations for all 12 pigs.
- `ped`: The data with columns `bv` and `dv` added.
"""
function ex_13_1()
    println("Example 13.1: additive and dominance effects solved separately")

    # 1. Pedigree, pens and records, one row per pig (records on pigs 5–12)
    (; ped, σ²a, σ²d, σ²e) = data_13_1()
    rec = dropmissing(ped, :ww)
    N = nrow(ped)

    # 2. Dominance relationships by Cockerham's rule: for i ≠ j,
    #    Dᵢⱼ = ¼(A_{sᵢsⱼ} A_{dᵢdⱼ} + A_{sᵢdⱼ} A_{dᵢsⱼ}), unknown parents
    #    contributing 0; Dᵢᵢ = 1 as no pig is inbred
    D = drm(ped)
    A = inv(Matrix(ainv(ped)))
    a(i, j) = i > 0 && j > 0 ? A[i, j] : 0.0
    s, d = ped.sire, ped.dam
    dij(i, j) = (a(s[i], s[j]) * a(d[i], d[j]) + a(s[i], d[j]) * a(d[i], s[j])) / 4
    De = [i == j ? 1.0 : dij(i, j) for i = 1:N, j = 1:N]
    check("D by Cockerham's rule = drm()", De, D; atol = 1e-12)
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

    # 3. MME (R = Iσ²e) as T'T + Σ: the same Z links a record to the pig's
    #    additive and its dominance effect, T = [X Z Z]
    y = rec.ww
    X, _ = incidence(rec.pen)
    Z = incidence(rec.pig, N)
    Ai = ainv(ped)
    T = [X Z Z]
    Σ = blockdiag(spzeros(2, 2), sparse(σ²e / σ²a * Ai), sparse(σ²e / σ²d * Di))
    m = mme(y, X, [RandomEffect(Z, Ai, σ²a), RandomEffect(Z, Di, σ²d)]; σ²e)
    check("mme() LHS = T'T + Σ", m.lhs, T'T + Σ; atol = 1e-10)

    # 4. Solve MME
    x = (T'T + Σ) \ (T'y)
    b, a, d = x[m.blocks[1]], x[m.blocks[2]], x[m.blocks[3]]
    ped.bv, ped.dv = a, d
    show_table(ped)
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

    (; b, a, d, ped)
end
