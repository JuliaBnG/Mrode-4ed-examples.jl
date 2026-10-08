"""
    ex_04_4() -> NamedTuple

Example 4.4: Animal model with unknown parent groups (UPG) using the
Quaas–Pollak (QP) transformation (Mrode & Pocrnic, 2023, Section 4.6;
Quaas & Pollak, 1981; Westell et al., 1988).

# Model Formulation
When ancestors have unknown parents, genetic groups can account for different
selection histories or genetic levels among base animals.
The explicit model with genetic groups is:
```math
y = Xb + ZQg + Za + e
```
where:
- ``g`` is the vector of fixed genetic group effects.
- ``Q`` is the matrix of expected group contributions to individual animals,
  computed recursively as ``Q = (I - P)^{-1} P_Q``.
- The total genetic merit (EBV including genetic group) is ``a^* = Qg + a``.

# Quaas–Pollak (QP) Transformation
Instead of adding explicit ``ZQg`` columns to the design matrix, the QP
transformation defines a modified inverse relationship matrix ``A^{*-1}`` (`ainv_upg`),
augmenting the equations directly by genetic groups:
```math
A^{*-1} = \\begin{bmatrix} I \\\\ -Q' \\end{bmatrix} A^{-1} \\begin{bmatrix} I & -Q
    \\end{bmatrix}
= \\begin{bmatrix} A^{-1} & -A^{-1}Q \\\\ -Q'A^{-1} & Q'A^{-1}Q \\end{bmatrix}
```
This yields solutions ``\\hat{g}`` and ``\\hat{a}^*`` directly without explicit ``Q``
multiplications.

# Returns
A `NamedTuple` with fields:
- `x`: Solution vector from the QP-transformed MME (group 1 constrained to 0).
- `astar`: ``\\hat{a} + Q\\hat{g}`` with ``\\hat{a}`` from the model without groups
  (Example 4.1), to compare with the QP solutions ``\\hat{a}^*``.
- `Q`: Matrix of genetic group contributions.
"""
function ex_04_4()
    println("Example 4.4: animal model with groups")

    # 1. Pedigree with genetic group codes:
    # unknown sires → group 1 (recoded as animal 9, coded -1)
    # unknown dams  → group 2 (recoded as animal 10, coded -2)
    ped = DataFrame(sire = [-1, -1, -1, 1, 3, 1, 4, 3], dam = [-2, -2, -2, -2, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]
    σ²a, σ²e = 20.0, 40.0

    # 2. Augmented inverse relationship matrix with groups (A*⁻¹)
    Ai = ainv_upg(ped)
    check(
        "A⁻¹ rows of the groups",
        Matrix(Ai)[9:10, :],
        [-0.5 -0.5 -0.5 0 0 0 0 0 0.75 0.75; -0.17 -0.5 -0.5 -0.67 0 0 0 0 0.75 1.08];
        atol = 5e-3,
    )

    # 3. Incidence matrices and MME setup
    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(calf, size(Ai, 1))  # 8 animals + 2 groups; no records on groups
    m = mme(wwg, X, [RandomEffect(Z, Ai, σ²a; name = "animal+group")]; σ²e)

    # 4. Solve MME with group 1 constrained to zero (benchmark group)
    x = solve_mme(m; zero = [m.blocks[2][9]])  # group 1 set to zero
    show_table(effect = ["male", "female", string.(1:8)..., "G9", "G10"], solution = x)
    check(
        "solutions (G9 = 0)",
        x,
        [
            5.458,
            4.313,
            -0.767,
            -0.923,
            -0.963,
            -1.268,
            -1.099,
            -0.728,
            -1.338,
            -0.768,
            0,
            -1.769,
        ],
    )

    # 5. Group contribution matrix Q and â* = â + Qĝ equivalence:
    # Compare with â from the model without groups (Example 4.1)
    Q = group_contributions(ped)
    check(
        "Q′ (= (TQ*)′)",
        Q',
        [0.5 0.5 0.5 0.25 0.5 0.5 0.375 0.5; 0.5 0.5 0.5 0.75 0.5 0.5 0.625 0.5];
        atol = 1e-12,
    )
    ped0 = DataFrame(sire = max.(ped.sire, 0), dam = max.(ped.dam, 0))
    m0 = mme(wwg, X, [RandomEffect(incidence(calf, 8), ainv(ped0), σ²a)]; σ²e)
    a0 = solve_mme(m0)[3:end]
    astar = a0 + Q * x[(end-1):end]
    show_table(animal = 1:8, a_groups = x[3:10], a_nogroup_plus_Qg = astar)

    # 6. Explicit group model: y = Xb + ZQg + Za + e (without QP transformation)
    mq = mme(
        wwg,
        [X Z[:, 1:8] * Q],
        [RandomEffect(incidence(calf, 8), ainv(ped0), σ²a)];
        σ²e,
    )
    xq = solve_mme(mq; zero = [3])
    check(
        "â + Qĝ from the explicit model = QP solutions",
        xq[mq.blocks[2]] + Q * xq[3:4],
        x[3:10];
        atol = 1e-8,
    )

    (; x, astar, Q)
end
