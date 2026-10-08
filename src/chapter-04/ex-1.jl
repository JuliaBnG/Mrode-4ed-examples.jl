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
- ``e`` is the vector of random residual errors, with ``\\mathrm{var}(e) = I \\sigma_e^2``.
- Variance components: ``\\sigma_a^2 = 20.0``, ``\\sigma_e^2 = 40.0``, giving
  variance ratio ``\\alpha = \\sigma_e^2 / \\sigma_a^2 = 2.0`` (heritability ``h^2 = 1/3``).

# Key Concepts Illustrated
1. **MME Formulation**: Mixed model equations setup using `incidence` and `ainv(ped)`.
2. **Accuracy of Evaluations** (Section 4.3.3):
   -  Prediction error variance: ``\\mathrm{PEV}_i = d_i \\sigma_e^2``, where ``d_i`` is the
     ``i``-th diagonal element of the animal block of ``C^{-1} = (\\mathrm{LHS})^{-1}``.
   - Reliability: ``r_i^2 = 1 - \\mathrm{PEV}_i / \\sigma_a^2 = 1 - d_i \\alpha``.
   - Standard error of prediction: ``\\mathrm{SEP}_i = \\sqrt{\\mathrm{PEV}_i}``.
3. **Partition of EBV** (Sections 4.3.1–4.3.2):
   - Decomposes each animal's EBV into Parent Average (PA), individual Yield
     Deviation (YD), and Progeny Contribution (PC):
     ``\\hat{a}_i = w_1 \\mathrm{PA}_i + w_2 \\mathrm{YD}_i + w_3 \\mathrm{PC}_i``.
   - Daughter Yield Deviation (DYD) of parents (e.g., sire 3).

# Returns
A `NamedTuple` with fields:
- `b`: Fixed effect solutions (sex).
- `a`: Vector of estimated breeding values (EBVs) for animals 1 to 8.
- `C`: Full inverse of the MME coefficient matrix ``C^{-1}``.
- `partition`: Results of EBV decomposition (PA, YD, PC, weights, and DYD).
"""
function ex_04_1()
    println("Example 4.1: animal model, pre-weaning gain of beef calves")

    # 1. Pedigree and phenotypic data
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]

    # 2. Variance components and variance ratio α
    σ²a = 20.0
    σ²e = 40.0
    α = σ²e / σ²a                               # α = (1 - h²) / h² = 2.0

    # 3. Incidence matrices and MME setup
    X, _ = incidence(sex; levels = ["M", "F"])  # fixed sex effects
    Z = incidence(calf, nrow(ped))              # random animal effects (all 8 animals)
    m = mme(wwg, X, [RandomEffect(Z, ainv(ped), σ²a; name = "animal")]; σ²e)

    # 4. Solve MME and extract solutions
    x = solve_mme(m)
    b, (a,) = solutions(m, x)
    a = vec(a)

    show_table(effect = ["male", "female", string.(1:8)...], solution = x)
    check("sex effects", b, [4.358, 3.404])
    check(
        "breeding values",
        a,
        [0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.249, 0.183],
    )

    # 5. Accuracy of evaluations (Section 4.3.3):
    #    PEV = C²² σ²e,  r² = 1 - dᵢ α,  SEP = √(PEV)
    C = lhs_inverse(m)
    d = diag(C)[m.blocks[2]]
    r² = 1 .- d .* α
    sep = sqrt.(d .* σ²e)

    show_table(animal = 1:8, d = d, r2 = r², r = sqrt.(r²), SEP = sep)
    check("diagonals of C⁻¹", d, [0.471, 0.492, 0.456, 0.428, 0.428, 0.442, 0.442, 0.422])
    check("reliabilities", r², [0.058, 0.016, 0.088, 0.144, 0.144, 0.116, 0.116, 0.156])
    check("SEP", sep, [4.341, 4.436, 4.271, 4.138, 4.138, 4.205, 4.205, 4.109]; atol = 3e-3)
    check("reliability() without inverting C", vec(reliability(m, 1)), r²; atol = 1e-10)

    # 6. Partition of EBV into PA, YD, and PC (Sections 4.3.1–4.3.2):
    #    EBV = w₁ PA + w₂ YD + w₃ PC, and progeny yield deviation (DYD)
    nrec = zeros(Int, 8)
    nrec[calf] .= 1
    yd = zeros(8)
    yd[calf] = wwg - X * b
    p = ebv_partition(ped.sire, ped.dam, α, nrec, yd, a)

    show_table(
        animal = 1:8,
        PA = p.PA,
        YD = p.YD,
        PC = p.PC,
        n1 = p.n1,
        n2 = p.n2,
        n3 = p.n3,
        EBV = p.n1 .* p.PA .+ p.n2 .* p.YD .+ p.n3 .* replace(p.PC, NaN => 0.0),
    )
    check("progeny yield deviation of sire 3", p.DYD[3], 0.059)

    (; b, a, C, partition = p)
end
