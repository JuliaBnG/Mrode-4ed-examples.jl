"""
    data_08() -> NamedTuple

Data of Table 8.1: Calf birth weights in 3 herds and 2 management pens,
pedigree for 14 animals (calves 5 to 14 with records), and (co)variance components.
`ped` holds all of it, one row per animal (IDs are row numbers; animals 1–4 have no
records); `calf`, `herd`, `pen` and `bw` are the records as vectors.
"""
function data_08()
    no4 = fill(missing, 4)
    ped = DataFrame(
        calf = 1:14,
        sire = [0, 0, 0, 0, 1, 3, 4, 3, 1, 3, 3, 8, 9, 3],
        dam = [0, 0, 0, 0, 2, 2, 6, 5, 6, 2, 7, 7, 2, 6],
        herd = [no4; [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]],
        pen = [no4; [1, 2, 2, 1, 1, 2, 2, 2, 1, 2]],
        bw = [no4; [35.0, 20.0, 25.0, 40.0, 42.0, 22.0, 35.0, 34.0, 20.0, 40.0]],
    )
    @assert ped.calf == 1:nrow(ped)
    rec = dropmissing(ped, :bw)
    G0 = [150.0 -40.0; -40.0 90.0]
    (;
        ped,
        calf = rec.calf,
        herd = rec.herd,
        pen = rec.pen,
        bw = rec.bw,
        G0,
        σ²pe = 40.0,
        σ²e = 350.0,
    )
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
1. **One table for the data**: pedigree, herd, pen and birth weight of all 14
   animals in one `DataFrame`; the direct, maternal and maternal pe solutions
   are added as columns.
2. **Correlated Direct and Maternal Random Effects**: the record of a calf is
   linked to its own direct effect by ``Z`` and to its dam's maternal effect by
   ``W``. With ``\\alpha = \\sigma_e^2 G_0^{-1}`` and
   ``\\lambda_{pe} = \\sigma_e^2 / \\sigma_{pe}^2``, the MME are
   ```math
   \\begin{bmatrix}
   X'X & X'Z & X'W & X'S \\\\
   Z'X & Z'Z + \\alpha_{11} A^{-1} & Z'W + \\alpha_{12} A^{-1} & Z'S \\\\
   W'X & W'Z + \\alpha_{21} A^{-1} & W'W + \\alpha_{22} A^{-1} & W'S \\\\
   S'X & S'Z & S'W & S'S + \\lambda_{pe} I
   \\end{bmatrix},
   ```
   written out explicitly and compared with
   `RandomEffect([Z, W], ainv(ped), G0)` in `mme`.
3. **Maternal Permanent Environment**: Associated with dams (animals 2, 5, 6, 7).

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `direct`, `maternal` and `pe` added.
- `b`: Fixed herd and pen effect solutions.
- `um`: Matrix of direct (col 1) and maternal (col 2) genetic solutions for all 14 animals.
- `pe`: Estimated maternal permanent environmental effects for dams.
"""
function ex_08_1()
    println("Example 8.1: maternal animal model, birth weight")

    # 1. Data: one row per animal, with the records of calves 5–14
    (; ped, G0, σ²pe, σ²e) = data_08()
    rec = dropmissing(ped, :bw)
    println(ped)

    # 2. Incidence matrices
    y = rec.bw
    X = [first(incidence(rec.herd)) first(incidence(rec.pen))]
    Z = incidence(rec.calf, nrow(ped))      # direct genetic effect (linked to calf)
    W = incidence(rec.dam, nrow(ped))       # maternal genetic effect (linked to dam)
    S, dams = incidence(rec.dam)            # maternal pe of dams 2, 5, 6, 7
    Ai = ainv(ped)

    # 3. MME written out (R = Iσ²e). Direct and maternal effects share A⁻¹ and
    #    are correlated, so their blocks are weighted by α = σ²e G₀⁻¹
    α = σ²e * inv(G0)
    λpe = σ²e / σ²pe
    check("variance ratios α", α, [2.647 1.176; 1.176 4.412])
    #! format: off
    lhs = [
        X'X  X'Z                   X'W                   X'S
        Z'X  Z'Z + α[1, 1] * Ai    Z'W + α[1, 2] * Ai    Z'S
        W'X  W'Z + α[2, 1] * Ai    W'W + α[2, 2] * Ai    W'S
        S'X  S'Z                   S'W                   S'S + λpe * I
    ]
    #! format: on
    rhs = [X'y; Z'y; W'y; S'y]

    m = mme(
        y,
        X,
        [
            RandomEffect([Z, W], Ai, G0; name = "direct+maternal"),
            RandomEffect(S, I, σ²pe; name = "pe"),
        ];
        σ²e,
    )
    check("mme() LHS = explicit form", m.lhs, lhs; atol = 1e-12)

    # 4. Solve MME (herd 1 set to zero)
    x = solve_mme(lhs, rhs; zero = [1])
    b = x[m.blocks[1]]
    um = reshape(x[m.blocks[2]], :, 2)      # rows: animals, cols: direct and maternal
    pe = x[m.blocks[3]]

    println("\nHerd and pen effects:")
    show_table(effect = ["herd " .* string.(1:3); "pen " .* string.(1:2)], solution = b)
    ped.direct, ped.maternal = um[:, 1], um[:, 2]
    ped.pe = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.pe[dams] = pe
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

    println("\nData with solutions:")
    show_table(ped)

    (; ped, b, um, pe)
end
