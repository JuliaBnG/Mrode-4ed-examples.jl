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
The MME, with ``\\lambda = \\sigma_e^2/\\sigma_a^2``, are then
```math
\\begin{bmatrix}
X'X & X'Z & 0 \\\\
Z'X & Z'Z + \\lambda A^{-1} & -\\lambda A^{-1}Q \\\\
0 & -\\lambda Q'A^{-1} & \\lambda Q'A^{-1}Q
\\end{bmatrix}
\\begin{bmatrix}\\hat{b} \\\\ \\hat{a}^* \\\\ \\hat{g}\\end{bmatrix} =
\\begin{bmatrix}X'y \\\\ Z'y \\\\ 0\\end{bmatrix},
```
which yield ``\\hat{g}`` and ``\\hat{a}^*`` directly without explicit ``Q``
multiplications in the design matrix.

# Key Concepts Illustrated
1. **Same data as Example 4.1**, in one `DataFrame`, with each unknown parent
   replaced by its group code: ``-1`` for unknown sires (group 1) and ``-2``
   for unknown dams (group 2). `ainv_upg(ped)` appends the groups as
   equations 9 and 10.
2. **``A^{*-1}`` from its blocks**: built explicitly from ``A^{-1}`` (groups
   treated as unknown parents) and ``Q``, and compared with `ainv_upg`.
3. **MME**, written out explicitly and compared with `mme`. The sex and group
   effects are confounded (each row of ``Q`` sums to 1), so group 1 is
   constrained to zero.
4. **Equivalence**: ``\\hat{a}^* = \\hat{a} + Q\\hat{g}`` from the explicit
   group model ``y = Xb + ZQg + Za + e``.

# Returns
A `NamedTuple` with fields:
- `ped`: The data with the group contributions `Q1`, `Q2`, and the solutions
  `astar` (``\\hat{a}^*``), `a` (``\\hat{a}`` of the explicit group model) and
  `a0_Qg` (the `astar` field below) added.
- `groups`: Table of the groups with their solutions ``\\hat{g}``.
- `x`: Solution vector from the QP-transformed MME (group 1 constrained to 0).
- `astar`: ``\\hat{a} + Q\\hat{g}`` with ``\\hat{a}`` from the model without groups
  (Example 4.1), to compare with the QP solutions ``\\hat{a}^*``.
- `Q`: Matrix of genetic group contributions.
"""
function ex_04_4()
    println("Example 4.4: animal model with groups")

    # 1. Pedigree and phenotypic data, as in Example 4.1, but with each unknown
    #    parent replaced by its group: unknown sires → -1, unknown dams → -2
    ped = DataFrame(
        calf = 1:8,
        sex = [missing, missing, missing, "M", "F", "F", "M", "M"],
        sire = [-1, -1, -1, 1, 3, 1, 4, 3],
        dam = [-2, -2, -2, -2, 2, 2, 5, 6],
        wwg = [missing, missing, missing, 4.5, 2.9, 3.9, 3.5, 5.0],
    )
    rec = dropmissing(ped, :wwg)                # the animals with records
    groups = DataFrame(group = [1, 2], code = [-1, -2], parent = ["sire", "dam"])
    n, ng = nrow(ped), nrow(groups)
    σ²a, σ²e = 20.0, 40.0
    λ = σ²e / σ²a

    # 2. Group contributions Q = (I - P)⁻¹ P_Q, and A⁻¹ with groups as unknown parents
    Q = group_contributions(ped)
    ped.Q1, ped.Q2 = Q[:, 1], Q[:, 2]
    ped0 = DataFrame(sire = max.(ped.sire, 0), dam = max.(ped.dam, 0))
    A0i = ainv(ped0)
    check(
        "Q′ (= (TQ*)′)",
        Q',
        [0.5 0.5 0.5 0.25 0.5 0.5 0.375 0.5; 0.5 0.5 0.5 0.75 0.5 0.5 0.625 0.5];
        atol = 1e-12,
    )

    # 3. A*⁻¹ = [A⁻¹ -A⁻¹Q; -Q'A⁻¹ Q'A⁻¹Q], and directly by ainv_upg (Henderson's rules)
    Ast = [
        A0i      -A0i*Q
        -Q'*A0i  Q'*A0i*Q
    ]
    Ai = ainv_upg(ped)
    check("A*⁻¹ from blocks = ainv_upg(ped)", Matrix(Ai), Ast; atol = 1e-12)
    check(
        "A⁻¹ rows of the groups",
        Matrix(Ai)[9:10, :],
        [-0.5 -0.5 -0.5 0 0 0 0 0 0.75 0.75; -0.17 -0.5 -0.5 -0.67 0 0 0 0 0.75 1.08];
        atol = 5e-3,
    )

    # 4. MME with groups as extra equations (no records on groups)
    y = rec.wwg
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Z = incidence(rec.calf, n)                  # records → animals 1:8
    lhs = [
        X'X           X'Z                spzeros(2, ng)
        Z'X           Z'Z + λ * A0i      -λ * A0i * Q
        spzeros(ng, 2)  -λ * Q' * A0i    λ * Q' * A0i * Q
    ]
    rhs = [X'y; Z'y; zeros(ng)]

    println("\nCoefficient matrix [X'X X'Z 0; Z'X Z'Z+λA⁻¹ -λA⁻¹Q; 0 -λQ'A⁻¹ λQ'A⁻¹Q]:")
    show(stdout, "text/plain", round.(Matrix(lhs); digits = 3))
    println()

    Zg = incidence(rec.calf, n + ng)            # 8 animals + 2 groups
    m = mme(y, X, [RandomEffect(Zg, Ai, σ²a; name = "animal+group")]; σ²e)
    check("mme() LHS = explicit QP form", m.lhs, lhs; atol = 1e-12)

    # 5. Solve with group 1 constrained to zero (benchmark group)
    g1 = m.blocks[2][n+1]
    x = solve_mme(lhs, rhs; zero = [g1])
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
    b = x[m.blocks[1]]
    ped.astar = x[m.blocks[2][1:n]]
    groups.g = x[m.blocks[2][(n+1):end]]
    println("\nSex effects:")
    show_table(sex = ["M", "F"], solution = b)
    println("\nGroups:")
    show_table(groups)

    # 6. Explicit group model y = Xb + ZQg + Za + e (without QP transformation):
    #    â* = â + Qĝ
    mq = mme(y, [X Z * Q], [RandomEffect(Z, A0i, σ²a; name = "animal")]; σ²e)
    xq = solve_mme(mq; zero = [3])
    ped.a = xq[mq.blocks[2]]
    check(
        "â + Qĝ from the explicit model = QP solutions",
        ped.a + Q * xq[3:4],
        ped.astar;
        atol = 1e-8,
    )

    # Compare with â from the model without groups (Example 4.1) plus Qĝ:
    # not the same, as groups were not fitted in that model
    m0 = mme(y, X, [RandomEffect(Z, A0i, σ²a; name = "animal")]; σ²e)
    astar = solve_mme(m0)[m0.blocks[2]] + Q * groups.g
    ped.a0_Qg = astar

    println("\nData with solutions:")
    show_table(ped)

    (; ped, groups, x, astar, Q)
end
