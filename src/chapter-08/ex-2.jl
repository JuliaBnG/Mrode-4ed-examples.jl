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

# Returns
A `NamedTuple` with fields:
- `b`: Fixed herd and pen solutions.
- `u`: Direct breeding values for all 14 animals.
- `mat`: Maternal breeding values for all 14 animals.
- `pe`: Maternal permanent environmental solutions.
"""
function ex_08_2()
    println("Example 8.2: reduced animal model with maternal effects")

    # 1. Load data and setup variables
    (; ped, calf, herd, pen, bw, G0, σ²pe, σ²e) = data_08()
    dam = ped.dam[calf]

    # 2. Setup RAM design for direct effect
    X = [first(incidence(herd)) first(incidence(pen))]
    Z1, rinv, parents, Api = ram_design(ped, calf, G0[1, 1], σ²e)
    col = zeros(Int, nrow(ped))
    col[parents] = eachindex(parents)
    Z2 = incidence(col[dam], length(parents))   # dams are parents
    S, _ = incidence(dam)

    # 3. Setup and solve reduced MME
    m = mme(bw, X, [RandomEffect([Z1, Z2], Api, G0), RandomEffect(S, I, σ²pe)]; Rinv = rinv)
    x = solve_mme(m; zero = [1])
    b, (um, pe) = solutions(m, x)

    check("R⁻¹", rinv, [fill(0.00286, 5); fill(0.00235, 5)]; atol = 1e-5)
    check("non-zeros in the coefficient matrix", nnz(m.lhs), 329)

    # 4. Back-solving direct effects û and maternal effects m̂ for non-parents (Eqns 8.8, 8.10)
    #    û from PA and Mendelian sampling; m̂ by genetic regression on direct Mendelian sampling
    yadj = bw - X * b - Z2 * um[:, 2] - S * vec(pe)
    u = ram_backsolve(ped, calf, yadj, um[:, 1], parents, G0[1, 1], σ²e)
    mat = fill(NaN, nrow(ped))
    mat[parents] = um[:, 2]
    for i = 1:nrow(ped)
        isnan(mat[i]) || continue
        s, d = ped.sire[i], ped.dam[i]
        pa(v) = 0.5 * ((s > 0 ? v[s] : 0.0) + (d > 0 ? v[d] : 0.0))
        mat[i] = pa(mat) + G0[1, 2] / G0[1, 1] * (u[i] - pa(u))
    end

    show_table(animal = 1:14, direct = u, maternal = mat)

    # 5. Verify equivalence with full animal model (Example 8.1)
    ref = ex_08_1()
    check("RAM = animal model, fixed effects", b, ref.b; atol = 1e-8)
    check("RAM = animal model, direct (all animals)", u, ref.um[:, 1]; atol = 1e-8)
    check("RAM = animal model, maternal (all animals)", mat, ref.um[:, 2]; atol = 1e-8)

    (; b, u, mat, pe)
end
