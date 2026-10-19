"""
    ex_05_1() -> NamedTuple

Example 5.1: Repeatability model for dairy cattle fat yield and daughter yield deviation
(DYD) (Mrode & Pocrnic, 2023, Sections 5.1–5.2).

# Model Formulation
For traits with repeated records per individual (e.g., lactations in dairy cattle):
```math
y = Xb + Za + Wpe + e
```
where:
- ``y`` is the vector of 10 lactation fat yield records (cows 4 to 8, parities 1 and 2).
- ``b`` is the vector of fixed herd-year-season (HYS 1 to 4) and parity (1 and 2) effects.
- ``a`` is the vector of random animal additive genetic effects (breeding values),
  with ``\\mathrm{var}(a) = A \\sigma_a^2``.
-  ``pe`` is the vector of permanent environmental effects for cows with records (cows 4 to
  8), with ``\\mathrm{var}(pe) = I \\sigma_{pe}^2``.
- ``e`` is the temporary environmental error, with ``\\mathrm{var}(e) = I \\sigma_e^2``.
-  Variance components: ``\\sigma_a^2 = 20.0``, ``\\sigma_{pe}^2 = 12.0``,
  ``\\sigma_e^2 = 28.0``. Total phenotypic variance ``\\sigma_P^2 = 60.0``; Repeatability
  ``r = (\\sigma_a^2 + \\sigma_{pe}^2) / \\sigma_P^2 = (20 + 12)/60 = 0.533``.

# Key Concepts Illustrated
1. **Two tables**: `ped` with one row per animal, and `rec` with one row per
   lactation. Per-cow quantities come from `groupby`/`combine` on `rec` and
   are joined onto `ped`; all animal-level results are columns of `ped`.
2. **Repeatability Model**: Fitting both additive genetic and permanent environmental
   effects. With ``R = I\\sigma_e^2`` the MME are, with
   ``\\lambda_a = \\sigma_e^2/\\sigma_a^2`` and ``\\lambda_{pe} = \\sigma_e^2/\\sigma_{pe}^2``,
   ```math
   \\begin{bmatrix}
   X'X & X'Z & X'W \\\\
   Z'X & Z'Z + \\lambda_a A^{-1} & Z'W \\\\
   W'X & W'Z & W'W + \\lambda_{pe} I
   \\end{bmatrix}
   \\begin{bmatrix}\\hat{b} \\\\ \\hat{a} \\\\ \\hat{pe}\\end{bmatrix} =
   \\begin{bmatrix}X'y \\\\ Z'y \\\\ W'y\\end{bmatrix},
   ```
   written out explicitly and compared with `mme`.
3. **Constraints**: ``X`` has 6 columns but rank 4. Besides the usual
   dependency between the HYS and parity levels, HYS 1 occurs only in
   parity 1 and HYS 3 only in parity 2. Two solutions are therefore set to
   zero (HYS 1 and 3, as in the book).
4. **Daughter Yield Deviation (DYD)** (Section 5.2.3):
   Records adjusted for fixed effects and permanent environmental effects:
   ``y^* = y - X\\hat{b} - W\\hat{pe}``.
   The daughter yield deviation of a sire is the weighted average of adjusted yields
   of his daughters, corrected for the genetic merit of the daughters' dams.

# Returns
A `NamedTuple` with fields:
- `ped`: The pedigree with columns `n`, `YD`, `ebv`, `pe`, `PA`, `PC` and `DYD` added.
- `rec`: The records with the adjusted yields `yadj` added.
- `b`: Fixed effect solutions (HYS and parity).
- `a`: Estimated breeding values for all 8 animals in the pedigree.
- `pe`: Estimated permanent environmental effects for cows 4 to 8.
- `partition`: Partition of sire EBVs and DYD (including DYD for sire 1).
"""
function ex_05_1()
    println("Example 5.1: repeatability model, fat yield")

    # 1. Pedigree, one row per animal (IDs are row numbers), and the
    #    lactation records, one row per record
    ped =
        DataFrame(id = 1:8, sire = [0, 0, 0, 1, 3, 1, 3, 1], dam = [0, 0, 0, 2, 2, 5, 4, 7])
    rec = DataFrame(
        cow = repeat(4:8; inner = 2),
        parity = repeat([1, 2], 5),
        hys = [1, 3, 1, 4, 2, 3, 1, 3, 2, 4],
        fy = [201.0, 280.0, 150.0, 200.0, 160.0, 190.0, 180.0, 250.0, 285.0, 300.0],
    )
    @assert ped.id == 1:nrow(ped)
    println(rec)

    # 2. Variance components and variance ratios
    σ²a = 20.0
    σ²pe = 12.0
    σ²e = 28.0
    λa = σ²e / σ²a                              # 1.4
    λpe = σ²e / σ²pe                            # 2.333

    # 3. Incidence matrices
    y = rec.fy
    X = [first(incidence(rec.hys)) first(incidence(rec.parity))]
    Z = incidence(rec.cow, nrow(ped))           # breeding values for all 8 animals
    W, cows = incidence(rec.cow)                # pe only for cows with records (cows 4–8)
    Ai = ainv(ped)
    println("\nrank(X) = ", rank(Matrix(X)), " with ", size(X, 2), " columns")

    # 4. MME, written out (R = Iσ²e)
    #! format: off
    lhs = [
        X'X  X'Z               X'W
        Z'X  Z'Z + λa * Ai     Z'W
        W'X  W'Z               W'W + λpe * I
    ]
    #! format: on
    rhs = [X'y; Z'y; W'y]

    m = mme(
        y,
        X,
        [RandomEffect(Z, Ai, σ²a; name = "animal"), RandomEffect(W, I, σ²pe; name = "pe")];
        σ²e,
    )
    check("mme() LHS = explicit form", m.lhs, lhs; atol = 1e-12)

    # 5. Solve MME with HYS 1 and 3 constrained to zero
    x = solve_mme(lhs, rhs; zero = [1, 3])      # HYS 1 and 3 set to zero
    b = x[m.blocks[1]]
    a = x[m.blocks[2]]
    pe = x[m.blocks[3]]

    println("\nFixed effects:")
    show_table(effect = ["HYS " .* string.(1:4); "parity " .* string.(1:2)], solution = b)
    check("HYS and parity", b, [0, 44.065, 0, 0.013, 175.472, 241.893])
    check("animals", a, [10.148, -3.084, -7.063, 13.581, -18.207, -18.387, 9.328, 24.194])
    check("permanent environment", pe, [8.417, -7.146, -17.229, -1.390, 17.347])

    ped.ebv = a
    ped.pe = Vector{Union{Missing,Float64}}(missing, nrow(ped))
    ped.pe[cows] = pe

    # 6. Yield deviations (Section 5.2.3): records adjusted for fixed effects and
    #    permanent environment, yadj = y - Xb - W pe, averaged per cow
    rec.yadj = y - X * b - W * pe
    ydev = combine(groupby(rec, :cow), nrow => :n, :yadj => mean => :YD)
    leftjoin!(ped, ydev; on = :id => :cow)
    ped.n = coalesce.(ped.n, 0)
    check(
        "YD of cows 4, 6 and 8",
        ped.YD[[4, 6, 8]],
        [23.4005, -38.486, 44.432];
        atol = 2e-3,
    )

    # 7. Partition of EBV, and DYD of the sires
    p = ebv_partition(ped.sire, ped.dam, λa, ped.n, coalesce.(ped.YD, 0.0), a)
    ped.PA = p.PA
    ped.PC = replace(p.PC, NaN => missing)      # missing: no progeny
    ped.DYD = replace(p.DYD, NaN => missing)
    check("DYD of sire 1", p.DYD[1], 23.552; atol = 2e-3)
    check(
        "EBV = n₁ PA + n₂ YD + n₃ PC",
        p.n1 .* p.PA .+ p.n2 .* p.YD .+ p.n3 .* replace(p.PC, NaN => 0.0),
        a;
        atol = 1e-10,
    )

    println("\nRecords with adjusted yields:")
    show_table(rec)
    println("\nPedigree with solutions:")
    show_table(ped)

    (; ped, rec, b, a, pe, partition = p)
end
