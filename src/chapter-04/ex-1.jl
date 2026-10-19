"""
    ex_04_1() -> NamedTuple

Example 4.1: Univariate animal model for pre-weaning gain (WWG) of beef calves
(Mrode & Pocrnic, 2023, Sections 4.1–4.3).

# Model Formulation
The standard animal model is:
```math
y = Xb + Za + e
```
where:
- ``y`` is the vector of pre-weaning gain records (5 calves: animals 4 to 8).
- ``b`` is the vector of fixed sex effects (Male, Female).
- ``a`` is the vector of random additive genetic effects (breeding values)
  for all 8 animals in the pedigree, with ``\\mathrm{var}(a) = A \\sigma_a^2``.
- ``e`` is the vector of random residual errors, with ``\\mathrm{var}(e) = R``.
- Variance components: ``\\sigma_a^2 = 20.0``, ``\\sigma_e^2 = 40.0``.

# Key Concepts Illustrated
1. **One table for the data**: pedigree, sex and records of all 8 animals
   in one `DataFrame`; the founders (1–3) have no records, and their sex
   is `missing` as it is not given in the book. `ainv(ped)` only uses the
   `sire` and `dam` columns, and animal IDs are the row numbers.
2. **MME, general form** (any ``R``, e.g. heterogeneous residual variances):
   ```math
   \\begin{bmatrix}
   X'R^{-1}X & X'R^{-1}Z \\\\
   Z'R^{-1}X & Z'R^{-1}Z + A^{-1}/\\sigma_a^2
   \\end{bmatrix}
   \\begin{bmatrix}\\hat{b} \\\\ \\hat{a}\\end{bmatrix} =
   \\begin{bmatrix}X'R^{-1}y \\\\ Z'R^{-1}y\\end{bmatrix}.
   ```
   Here ``\\mathrm{PEV} = C^{-1}`` directly.
3. **MME, ``R = I\\sigma_e^2``**: multiplying through by ``\\sigma_e^2`` gives
   the form printed in the book, with ``\\lambda = \\sigma_e^2/\\sigma_a^2 = 2``:
   ```math
   \\begin{bmatrix}
   X'X & X'Z \\\\
   Z'X & Z'Z + \\lambda A^{-1}
   \\end{bmatrix}
   \\begin{bmatrix}\\hat{b} \\\\ \\hat{a}\\end{bmatrix} =
   \\begin{bmatrix}X'y \\\\ Z'y\\end{bmatrix}.
   ```
   This needs a single ratio ``\\lambda``, i.e. homogeneous residuals.
   Both forms are written out explicitly and compared with `mme`.
4. **Accuracy of Evaluations** (Section 4.3.3):
   - Prediction error variance: ``\\mathrm{PEV}_i = d_i \\sigma_e^2``, where ``d_i`` is the
     ``i``-th diagonal element of the animal block of ``C^{-1}`` of the ``\\lambda`` form.
   - Reliability: ``r_i^2 = 1 - \\mathrm{PEV}_i / \\sigma_a^2 = 1 - d_i \\lambda``.
   - Standard error of prediction: ``\\mathrm{SEP}_i = \\sqrt{\\mathrm{PEV}_i}``.
5. **Partition of EBV** (Sections 4.3.1–4.3.2):
   - Decomposes each animal's EBV into Parent Average (PA), individual Yield
     Deviation (YD), and Progeny Contribution (PC):
     ``\\hat{a}_i = w_1 \\mathrm{PA}_i + w_2 \\mathrm{YD}_i + w_3 \\mathrm{PC}_i``.
   - Daughter Yield Deviation (DYD) of parents (e.g., sire 3).

All animal-level results are added as columns of `ped`.

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `ebv`, `d`, `r2`, `sep`, `PA`, `YD` and `PC` added.
- `b`: Fixed effect solutions (sex).
- `a`: Vector of estimated breeding values (EBVs) for animals 1 to 8.
- `C`: Full inverse of the MME coefficient matrix ``C^{-1}`` (``\\lambda`` form).
- `partition`: Results of EBV decomposition (PA, YD, PC, weights, and DYD).
"""
function ex_04_1()
    println("Example 4.1: animal model, pre-weaning gain of beef calves")

    # 1. Pedigree and phenotypic data, one row per animal.
    #    Animal IDs are row numbers; parents precede their progeny.
    ped = DataFrame(
        calf = 1:8,
        sex = [missing, missing, missing, "M", "F", "F", "M", "M"],
        sire = [0, 0, 0, 1, 3, 1, 4, 3],
        dam = [0, 0, 0, 0, 2, 2, 5, 6],
        wwg = [missing, missing, missing, 4.5, 2.9, 3.9, 3.5, 5.0],
    )
    @assert ped.calf == 1:nrow(ped)
    rec = dropmissing(ped, :wwg)                # the animals with records
    println(ped)

    # 2. Variance components
    σ²a = 20.0
    σ²e = 40.0

    # 3. Incidence matrices and A⁻¹
    y = rec.wwg
    X, _ = incidence(rec.sex; levels = ["M", "F"])  # fixed sex effects
    Z = incidence(rec.calf, nrow(ped))              # random animal effects (all 8 animals)
    Ai = ainv(ped)                                  # uses only ped.sire and ped.dam

    # 4a. MME, general form: any R, e.g. heterogeneous residual variances.
    #     Here R = Iσ²e, so R⁻¹ is a diagonal of 1/σ²e.
    Ri = Diagonal(fill(1 / σ²e, length(y)))
    #! format: off
    lhs_R = [
        X'Ri*X  X'Ri*Z
        Z'Ri*X  Z'Ri*Z + Ai / σ²a
    ]
    #! format: on
    rhs_R = [X'Ri*y; Z'Ri*y]
    x_R = lhs_R \ rhs_R

    # 4b. R = Iσ²e: multiply the general form by σ²e. Only a single
    #     variance ratio λ is left, as printed in the book.
    λ = σ²e / σ²a                               # λ = (1 - h²) / h² = 2.0
    #! format: off
    lhs = [
        X'X  X'Z
        Z'X  Z'Z + λ * Ai
    ]
    #! format: on
    rhs = [X'y; Z'y]
    x = lhs \ rhs

    println("\nCoefficient matrix [X'X X'Z; Z'X Z'Z + λA⁻¹]:")
    show(stdout, "text/plain", round.(Matrix(lhs); digits = 3))
    println()
    println("\nRight-hand side [X'y; Z'y]: ", round.(rhs; digits = 3))
    check("general R⁻¹ form = λ form", x_R, x; atol = 1e-10)

    # 4c. The same equations built by mme(): with σ²e (λ form), and with
    #     Rinv (general form, a vector of diagonal weights or a matrix).
    m = mme(y, X, [RandomEffect(Z, Ai, σ²a; name = "animal")]; σ²e)
    m_R = mme(y, X, [RandomEffect(Z, Ai, σ²a; name = "animal")]; Rinv = diag(Ri))
    check("mme(; σ²e) LHS = [X'X X'Z; Z'X Z'Z+λA⁻¹]", m.lhs, lhs; atol = 1e-12)
    check("mme(; Rinv) LHS = general form", m_R.lhs, lhs_R; atol = 1e-12)

    b = x[m.blocks[1]]
    a = x[m.blocks[2]]
    ped.ebv = a
    println("\nSex effects:")
    show_table(sex = ["M", "F"], solution = b)
    check("sex effects", b, [4.358, 3.404])
    check(
        "breeding values",
        a,
        [0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.249, 0.183],
    )

    # 5. Accuracy of evaluations (Section 4.3.3):
    #    λ form: PEV = dᵢ σ²e, r² = 1 - dᵢ λ;  general form: PEV = C⁻¹ directly.
    C = lhs_inverse(m)
    ped.d = diag(C)[m.blocks[2]]
    ped.r2 = 1 .- ped.d .* λ
    ped.sep = sqrt.(ped.d .* σ²e)
    pev_R = diag(lhs_inverse(m_R))[m_R.blocks[2]]

    check(
        "diagonals of C⁻¹",
        ped.d,
        [0.471, 0.492, 0.456, 0.428, 0.428, 0.442, 0.442, 0.422],
    )
    check("reliabilities", ped.r2, [0.058, 0.016, 0.088, 0.144, 0.144, 0.116, 0.116, 0.156])
    check(
        "SEP",
        ped.sep,
        [4.341, 4.436, 4.271, 4.138, 4.138, 4.205, 4.205, 4.109];
        atol = 3e-3,
    )
    check("PEV of general form = dᵢ σ²e", pev_R, ped.d .* σ²e; atol = 1e-10)
    check("reliability() without inverting C", vec(reliability(m, 1)), ped.r2; atol = 1e-10)

    # 6. Partition of EBV into PA, YD, and PC (Sections 4.3.1–4.3.2):
    #    EBV = w₁ PA + w₂ YD + w₃ PC, and progeny yield deviation (DYD)
    nrec = Int.(.!ismissing.(ped.wwg))
    yd = zeros(nrow(ped))
    yd[rec.calf] = y - X * b
    p = ebv_partition(ped.sire, ped.dam, λ, nrec, yd, a)
    ped.PA = p.PA
    ped.YD = ifelse.(nrec .> 0, p.YD, missing)     # missing: no record
    ped.PC = replace(p.PC, NaN => missing)          # missing: no progeny

    println("\nWeights of the partition, and EBV rebuilt from it:")
    show_table(
        calf = ped.calf,
        n1 = p.n1,
        n2 = p.n2,
        n3 = p.n3,
        EBV = p.n1 .* p.PA .+ p.n2 .* p.YD .+ p.n3 .* replace(p.PC, NaN => 0.0),
    )
    check("progeny yield deviation of sire 3", p.DYD[3], 0.059)

    println("\nData with solutions:")
    show_table(ped)

    (; ped, b, a, C, partition = p)
end
