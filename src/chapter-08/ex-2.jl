"""
    ex_08_2() -> NamedTuple

Example 8.2: Reduced Animal Model (RAM) with maternal effects
(Mrode & Pocrnic, 2023, Section 8.3; Quaas & Pollak, 1980).

# Model Formulation
In RAM with maternal effects, only parents are included in the random equations:
```math
y = Xb + Z_1 u_p + Z_2 m_p + S pe_{\\mathrm{mat}} + e^*
```
where:
-  Direct effects of non-parents are absorbed into their parents' direct effects (via
  ``Z_1``).
-  Maternal effects of dams are included directly (dams are always parents, represented in
  ``Z_2``).
- After solving the reduced MME:
  1. Direct breeding values ``\\hat{u}`` of non-parents are back-solved from parent averages
     and Mendelian sampling deviations.
  2.  Maternal breeding values ``\\hat{m}`` of non-parents are back-solved using the
     regression of maternal Mendelian sampling on direct Mendelian sampling:
     ``\\hat{m}_i = \\mathrm{PA}(m) + \\frac{\\sigma_{um}}{\\sigma_u^2}
      [\\hat{u}_i - \\mathrm{PA}(u)]``.

# Key Concepts Illustrated
1. **Same data as Example 8.1**, one row per animal; parents, Mendelian-sampling
   factors and all solutions are added as columns.
2. **Heterogeneous residuals**: records of non-parents carry the Mendelian
   sampling of their direct effect, ``\\sigma_e^2 + m_i g_{11}``, so the reduced
   MME are in the general ``R^{-1}`` form, written out explicitly and compared
   with `mme(...; Rinv)`.
3. **Back-solving** direct (Eqn 8.8) and maternal (Eqn 8.10) effects of
   non-parents, computed explicitly and compared with `ram_backsolve`.
4. **Same solutions as the full animal model** of Example 8.1.

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `parent`, `m`, `direct`, `maternal` and `pe` added.
- `b`: Fixed herd and pen solutions.
- `u`: Direct breeding values for all 14 animals.
- `mat`: Maternal breeding values for all 14 animals.
- `pe`: Maternal permanent environmental solutions.
"""
function ex_08_2()
    println("Example 8.2: reduced animal model with maternal effects")

    # 1. Data of Example 8.1: one row per animal, with the records of calves 5–14
    (; ped, G0, σ²pe, σ²e) = data_08()
    rec = dropmissing(ped, :bw)
    y = rec.bw

    # 2. Parents, and the Mendelian-sampling variance factor
    #    mᵢ = 1 - ¼(1 + F_s) - ¼(1 + F_d), a term for each known parent
    ped.parent = in(Set([ped.sire; ped.dam])).(ped.calf)
    f1 = nrm_diag(ped)                          # 1 + F, the diagonal of A
    ped.m = [
        1 - (s > 0 ? f1[s] / 4 : 0.0) - (d > 0 ? f1[d] / 4 : 0.0) for
        (s, d) in zip(ped.sire, ped.dam)
    ]

    # 3. RAM design for the direct effect: W₁ (records × parents), R⁻¹ with
    #    1/σ²e for parents and 1/(σ²e + m g₁₁) for non-parents, and A_p⁻¹
    X = [first(incidence(rec.herd)) first(incidence(rec.pen))]
    Z1, rinv, parents, Api = ram_design(ped, rec.calf, G0[1, 1], σ²e)
    check("parents", parents, findall(ped.parent); atol = 0)
    check("R⁻¹", rinv, [fill(0.00286, 5); fill(0.00235, 5)]; atol = 1e-5)
    #    Maternal effects belong to the dams, which are always parents
    col = zeros(Int, nrow(ped))
    col[parents] = eachindex(parents)
    Z2 = incidence(col[rec.dam], length(parents))
    S, dams = incidence(rec.dam)

    # 4. MME in the general form, since R ≠ Iσ²e
    Ri = Diagonal(rinv)
    Gi = inv(G0)
    #! format: off
    lhs = [
        X'Ri*X   X'Ri*Z1                   X'Ri*Z2                   X'Ri*S
        Z1'Ri*X  Z1'Ri*Z1 + Gi[1, 1] * Api Z1'Ri*Z2 + Gi[1, 2] * Api Z1'Ri*S
        Z2'Ri*X  Z2'Ri*Z1 + Gi[2, 1] * Api Z2'Ri*Z2 + Gi[2, 2] * Api Z2'Ri*S
        S'Ri*X   S'Ri*Z1                   S'Ri*Z2                   S'Ri*S + I / σ²pe
    ]
    #! format: on
    rhs = [X'Ri*y; Z1'Ri*y; Z2'Ri*y; S'Ri*y]

    m = mme(y, X, [RandomEffect([Z1, Z2], Api, G0), RandomEffect(S, I, σ²pe)]; Rinv = rinv)
    check("mme(; Rinv) LHS = general form", m.lhs, lhs; atol = 1e-12)
    check("non-zeros in the coefficient matrix", nnz(m.lhs), 329)

    x = solve_mme(lhs, rhs; zero = [1])         # herd 1 set to zero
    b = x[m.blocks[1]]
    um = reshape(x[m.blocks[2]], :, 2)          # parents × (direct, maternal)
    pe = x[m.blocks[3]]

    # 5. Back-solving non-parents, in pedigree order (parents precede progeny):
    #    direct (Eqn 8.8):   ûᵢ = PA(u) + kᵢ (yᵢ - x'b̂ - m̂_d - p̂e_d - PA(u)),
    #                        kᵢ = mᵢ g₁₁ / (σ²e + mᵢ g₁₁)
    #    maternal (Eqn 8.10): m̂ᵢ = PA(m) + g₁₂/g₁₁ (ûᵢ - PA(u))
    ped.direct = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.maternal = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.direct[parents], ped.maternal[parents] = um[:, 1], um[:, 2]
    ped.pe = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.pe[dams] = pe
    yadj = y - X * b - Z2 * um[:, 2] - S * pe   # records adjusted for b, dam's m and pe
    function pa(v, i)                           # parent average; unknown parents count 0
        s, d = ped.sire[i], ped.dam[i]
        ((s > 0 ? v[s] : 0.0) + (d > 0 ? v[d] : 0.0)) / 2
    end
    for (j, i) in enumerate(rec.calf)
        ped.parent[i] && continue
        k = ped.m[i] * G0[1, 1] / (σ²e + ped.m[i] * G0[1, 1])
        msu = k * (yadj[j] - pa(ped.direct, i))   # Mendelian sampling, direct
        ped.direct[i] = pa(ped.direct, i) + msu
        ped.maternal[i] = pa(ped.maternal, i) + G0[1, 2] / G0[1, 1] * msu
    end
    u = Vector{Float64}(ped.direct)
    mat = Vector{Float64}(ped.maternal)
    check("direct of animal 10 (book: -1.055)", u[10], -1.055)
    check(
        "ram_backsolve() = explicit back-solving (direct)",
        ram_backsolve(ped, rec.calf, yadj, um[:, 1], parents, G0[1, 1], σ²e),
        u;
        atol = 1e-12,
    )

    # 6. The full animal model of Example 8.1 gives the same solutions
    Z = incidence(rec.calf, nrow(ped))
    W = incidence(rec.dam, nrow(ped))
    Ai = ainv(ped)
    m_a = mme(y, X, [RandomEffect([Z, W], Ai, G0), RandomEffect(S, I, σ²pe)]; σ²e)
    x_a = solve_mme(m_a; zero = [1])
    um_a = reshape(x_a[m_a.blocks[2]], :, 2)
    check("RAM = animal model, fixed effects", b, x_a[m_a.blocks[1]]; atol = 1e-8)
    check("RAM = animal model, direct (all animals)", u, um_a[:, 1]; atol = 1e-8)
    check("RAM = animal model, maternal (all animals)", mat, um_a[:, 2]; atol = 1e-8)

    println("\nHerd and pen effects:")
    show_table(effect = ["herd " .* string.(1:3); "pen " .* string.(1:2)], solution = b)
    println("\nData with solutions:")
    show_table(ped)

    (; ped, b, u, mat, pe)
end
