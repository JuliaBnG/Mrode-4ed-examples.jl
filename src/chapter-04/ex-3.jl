"""
    ex_04_3() -> NamedTuple

Example 4.3: Reduced Animal Model (RAM) for pre-weaning gain of beef calves
(Mrode & Pocrnic, 2023, Section 4.5; Quaas & Pollak, 1980).

# Model Formulation
The breeding value of a non-parent ``i`` is its parent average plus a
Mendelian sampling deviation, ``a_i = \\frac{1}{2}(a_{s_i} + a_{d_i}) + m_i``.
Moving the Mendelian sampling of non-parents into the residual gives
```math
y = Xb + W a_p + e^*
```
where:
- Only parents (animals 1 to 6) are directly evaluated in the MME.
- ``W`` maps records to parents: a row has a ``1`` for a parent's own record,
  and ``\\frac{1}{2}`` for the sire and dam of a non-parent.
- ``\\mathrm{var}(a_p) = A_p \\sigma_a^2``, with ``A_p`` the relationships
  among parents.
- Residual variance is ``\\sigma_e^2`` for records of parents and
  ``\\sigma_e^2 + m_i \\sigma_a^2`` for records of non-parents, where
  ``m_i = 1 - \\frac{1}{4}(1 + F_{s_i}) - \\frac{1}{4}(1 + F_{d_i})``
  (a term only for each known parent) is the Mendelian-sampling variance
  factor, ``\\frac{1}{2}`` here.

# Key Concepts Illustrated
1. **Same data as Example 4.1**: one `DataFrame` with pedigree, records and
   all animal-level results.
2. **Heterogeneous residuals**: ``R`` is diagonal but not ``I\\sigma_e^2``,
   so there is no single variance ratio ``\\lambda``, and the MME must be
   written in the general form:
   ```math
   \\begin{bmatrix}
   X'R^{-1}X & X'R^{-1}W \\\\
   W'R^{-1}X & W'R^{-1}W + A_p^{-1}/\\sigma_a^2
   \\end{bmatrix}
   \\begin{bmatrix}\\hat{b} \\\\ \\hat{a}_p\\end{bmatrix} =
   \\begin{bmatrix}X'R^{-1}y \\\\ W'R^{-1}y\\end{bmatrix}.
   ```
   It is written out explicitly and compared with `mme(...; Rinv)`.
3. **Back-solving** non-parents (Eqn 4.26):
   ```math
   \\hat{a}_i = \\mathrm{PA}_i + k_i \\left( y_i - x_i' \\hat{b} - \\mathrm{PA}_i \\right),
   \\quad \\mathrm{PA}_i = \\tfrac{1}{2}(\\hat{a}_{s_i} + \\hat{a}_{d_i}),
   \\quad k_i = \\frac{m_i \\sigma_a^2}{\\sigma_e^2 + m_i \\sigma_a^2},
   ```
   so ``k_i = 10/50 = 0.2`` here. Computed explicitly and compared with
   `ram_backsolve`.
4. **Same solutions as the full animal model** of Example 4.1, from 8
   instead of 10 equations.

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `parent`, `m`, `rinv`, `PA`, `k` and `ebv` added.
- `b`: Fixed sex effects solutions.
- `a`: Full vector of EBVs for all 8 animals (parents + back-solved non-parents).
- `ap`: Parent breeding value solutions from RAM MME.
- `rinv`: Diagonal elements of ``R^{-1}``.
"""
function ex_04_3()
    println("Example 4.3: reduced animal model")

    # 1. Pedigree and phenotypic data, as in Example 4.1
    ped = DataFrame(
        calf = 1:8,
        sex = [missing, missing, missing, "M", "F", "F", "M", "M"],
        sire = [0, 0, 0, 1, 3, 1, 4, 3],
        dam = [0, 0, 0, 0, 2, 2, 5, 6],
        wwg = [missing, missing, missing, 4.5, 2.9, 3.9, 3.5, 5.0],
    )
    rec = dropmissing(ped, :wwg)                # the animals with records
    σ²a, σ²e = 20.0, 40.0

    # 2. Parents, and the Mendelian-sampling variance factor
    #    mᵢ = 1 - ¼(1 + F_s) - ¼(1 + F_d), a term for each known parent
    ped.parent = in(Set([ped.sire; ped.dam])).(ped.calf)
    f1 = nrm_diag(ped)                          # 1 + F, the diagonal of A
    ped.m = [
        1 - (s > 0 ? f1[s] / 4 : 0.0) - (d > 0 ? f1[d] / 4 : 0.0) for
        (s, d) in zip(ped.sire, ped.dam)
    ]

    # 3. RAM design: W (records × parents), diagonal of R⁻¹, the parents and A_p⁻¹
    W, rinv, parents, Api = ram_design(ped, rec.calf, σ²a, σ²e)
    ped.rinv = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.rinv[rec.calf] = rinv
    check("parents", parents, findall(ped.parent); atol = 0)
    check(
        "R⁻¹: 1/σ²e (parents), 1/(σ²e + mσ²a)",
        rinv,
        @. ifelse(ped.parent[rec.calf], 1 / σ²e, 1 / (σ²e + ped.m[rec.calf] * σ²a))
    )
    println("\nW (rows: records of calves $(rec.calf); columns: parents $parents):")
    show(stdout, "text/plain", Matrix(W))
    println()

    # 4. MME in the general form, since R ≠ Iσ²e
    y = rec.wwg
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Ri = Diagonal(rinv)
    #! format: off
    lhs = [
        X'Ri*X  X'Ri*W
        W'Ri*X  W'Ri*W + Api / σ²a
    ]
    #! format: on
    rhs = [X'Ri*y; W'Ri*y]
    x = lhs \ rhs

    println("\nCoefficient matrix [X'R⁻¹X X'R⁻¹W; W'R⁻¹X W'R⁻¹W + A_p⁻¹/σ²a]:")
    show(stdout, "text/plain", round.(Matrix(lhs); digits = 4))
    println()
    println("\nRight-hand side [X'R⁻¹y; W'R⁻¹y]: ", round.(rhs; digits = 4))

    m = mme(y, X, [RandomEffect(W, Api, σ²a; name = "parents")]; Rinv = rinv)
    check("mme(; Rinv) LHS = general form", m.lhs, lhs; atol = 1e-12)
    check("R⁻¹", rinv, [0.025, 0.025, 0.025, 0.020, 0.020])
    check("W'R⁻¹y", W' * Ri * y, [0, 0, 0.050, 0.148, 0.107, 0.148])
    check(
        "sex and parent solutions",
        x,
        [4.358, 3.404, 0.098, -0.019, -0.041, -0.009, -0.186, 0.177],
    )
    check("non-zeros in the coefficient matrix", nnz(m.lhs), 38)

    b = x[m.blocks[1]]
    ap = x[m.blocks[2]]
    println("\nSex effects:")
    show_table(sex = ["M", "F"], solution = b)

    # 5. Back-solve non-parents (Eqn 4.26): âᵢ = PAᵢ + kᵢ (yᵢ - xᵢ'b̂ - PAᵢ)
    ped.ebv = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.ebv[parents] = ap
    ped.PA = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.k = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    yadj = y - X * b
    for (j, i) in enumerate(rec.calf)           # parents precede progeny
        ped.parent[i] && continue
        s, d = ped.sire[i], ped.dam[i]
        ped.PA[i] = ((s > 0 ? ped.ebv[s] : 0.0) + (d > 0 ? ped.ebv[d] : 0.0)) / 2
        ped.k[i] = ped.m[i] * σ²a / (σ²e + ped.m[i] * σ²a)
        ped.ebv[i] = ped.PA[i] + ped.k[i] * (yadj[j] - ped.PA[i])
    end
    a = Vector{Float64}(ped.ebv)
    check("back-solved non-parents 7 and 8", a[7:8], [-0.249, 0.183])
    check(
        "ram_backsolve() = explicit back-solving",
        ram_backsolve(ped, rec.calf, yadj, ap, parents, σ²a, σ²e),
        a;
        atol = 1e-12,
    )

    # 6. The full animal model of Example 4.1 gives the same solutions
    Z = incidence(rec.calf, nrow(ped))
    m_a = mme(y, X, [RandomEffect(Z, ainv(ped), σ²a; name = "animal")]; σ²e)
    check("RAM = full animal model (10 equations)", solve_mme(m_a), [b; a]; atol = 1e-10)

    println("\nData with solutions:")
    show_table(ped)

    (; ped, b, a, ap, rinv)
end
