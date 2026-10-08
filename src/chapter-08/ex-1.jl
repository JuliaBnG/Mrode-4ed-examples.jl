"""
    data_08() -> NamedTuple

Data of Table 8.1: Calf birth weights in 3 herds and 2 management pens,
pedigree for 14 animals (calves 5 to 14 with records), and (co)variance components.
"""
function data_08()
    ped = DataFrame(
        sire = [0, 0, 0, 0, 1, 3, 4, 3, 1, 3, 3, 8, 9, 3],
        dam = [0, 0, 0, 0, 2, 2, 6, 5, 6, 2, 7, 7, 2, 6],
    )
    calf = 5:14
    herd = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    pen = [1, 2, 2, 1, 1, 2, 2, 2, 1, 2]
    bw = [35.0, 20.0, 25.0, 40.0, 42.0, 22.0, 35.0, 34.0, 20.0, 40.0]
    G0 = [150.0 -40.0; -40.0 90.0]
    (; ped, calf, herd, pen, bw, G0, σ²pe = 40.0, σ²e = 350.0)
end

"""
    ex_08_1() -> NamedTuple

Example 8.1: Animal model for a maternal trait (birth weight of beef calves)
(Mrode & Pocrnic, 2023, Sections 8.1–8.2).

# Model Formulation
Phenotypes for maternal traits are influenced by the calf's own genes (direct genetic
effect) and the dam's maternal environment (maternal genetic and maternal permanent
environmental effects):
```math
y = Xb + Z u + W m + S pe_{\\mathrm{mat}} + e
```
where:
- ``b`` is the vector of fixed herd and pen effects.
-  ``u`` is the direct additive genetic effect of the calf, with
  ``\\mathrm{var}(u) = A \\sigma_u^2``.
-  ``m`` is the maternal additive genetic effect of the dam, with
  ``\\mathrm{var}(m) = A \\sigma_m^2``.
-  Direct and maternal genetic effects are correlated:
  ``\\mathrm{cov}(u, m) = \\sigma_{um}``, so
  ``\\mathrm{var}\\begin{pmatrix} u \\\\ m \\end{pmatrix} = G_0 \\otimes A`` with
  ``G_0 = \\begin{bmatrix} 150.0 & -40.0 \\\\ -40.0 & 90.0 \\end{bmatrix}``.
-  ``pe_{\\mathrm{mat}}`` is the maternal permanent environmental effect of dams with
  recorded progeny, with ``\\mathrm{var}(pe_{\\mathrm{mat}}) = I \\sigma_{pe}^2``
  (``\\sigma_{pe}^2 = 40.0``).
- ``e`` is the temporary environmental error, with ``\\sigma_e^2 = 350.0``.

# Key Concepts Illustrated
1.  **Correlated Direct and Maternal Random Effects**: Modeled jointly using
   `RandomEffect([Z, W], ainv(ped), G0)`.
2. **Maternal Permanent Environment**: Associated with dams (animals 2, 5, 6, 7).

# Returns
A `NamedTuple` with fields:
- `b`: Fixed herd and pen effect solutions.
- `um`: Matrix of direct (col 1) and maternal (col 2) genetic solutions for all 14 animals.
- `pe`: Estimated maternal permanent environmental effects for dams.
"""
function ex_08_1()
    println("Example 8.1: maternal animal model, birth weight")

    # 1. Load data and setup variables
    (; ped, calf, herd, pen, bw, G0, σ²pe, σ²e) = data_08()
    dam = ped.dam[calf]

    # 2. Incidence matrices
    X = [first(incidence(herd)) first(incidence(pen))]
    Z = incidence(calf, nrow(ped))          # direct genetic effect (linked to calf)
    W = incidence(dam, nrow(ped))           # maternal genetic effect (linked to dam)
    S, dams = incidence(dam)                # maternal pe of dams 2, 5, 6, 7

    # 3. Setup MME
    m = mme(
        bw,
        X,
        [
            RandomEffect([Z, W], ainv(ped), G0; name = "direct+maternal"),
            RandomEffect(S, I, σ²pe; name = "pe"),
        ];
        σ²e,
    )
    check("variance ratios α", σ²e * inv(G0), [2.647 1.176; 1.176 4.412])

    # 4. Solve MME (herd 1 set to zero)
    x = solve_mme(m; zero = [1])
    b, (um, pe) = solutions(m, x)

    show_table(animal = 1:14, direct = um[:, 1], maternal = um[:, 2])
    check("herd and pen", b, [0, 3.386, 1.434, 34.540, 27.691])
    check(
        "direct",
        um[:, 1],
        [
            0.564,
            -1.244,
            1.165,
            -0.484,
            0.630,
            -0.859,
            -1.156,
            1.917,
            -0.553,
            -1.055,
            0.385,
            0.863,
            -2.980,
            1.751,
        ],
    )
    check(
        "maternal",
        um[:, 2],
        [
            0.262,
            -1.583,
            0.736,
            0.586,
            -0.507,
            0.841,
            1.299,
            -0.158,
            0.660,
            -0.153,
            0.916,
            0.442,
            0.093,
            0.362,
        ],
    )
    check("permanent environment (dams $(dams))", pe, [-1.701, 0.415, 0.825, 0.461])
    check("non-zeros in the coefficient matrix", nnz(m.lhs), 429)

    (; b, um, pe)
end
