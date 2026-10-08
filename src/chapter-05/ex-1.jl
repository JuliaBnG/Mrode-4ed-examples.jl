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
1.  **Repeatability Model**: Fitting both additive genetic and permanent environmental
   effects.
2. **Daughter Yield Deviation (DYD)** (Section 5.2.3):
   Records adjusted for fixed effects and permanent environmental effects:
   ``y^* = y - X\\hat{b} - W\\hat{pe}``.
   The daughter yield deviation of a sire is the weighted average of adjusted yields
   of his daughters, corrected for the genetic merit of the daughters' dams.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed effect solutions (HYS and parity).
- `a`: Estimated breeding values for all 8 animals in the pedigree.
- `pe`: Estimated permanent environmental effects for cows 4 to 8.
- `partition`: Partition of sire EBVs and DYD (including DYD for sire 1).
"""
function ex_05_1()
    println("Example 5.1: repeatability model, fat yield")

    # 1. Pedigree and repeated lactation records
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 3, 1], dam = [0, 0, 0, 2, 2, 5, 4, 7])
    cow = repeat(4:8; inner = 2)
    parity = repeat([1, 2], 5)
    hys = [1, 3, 1, 4, 2, 3, 1, 3, 2, 4]
    fy = [201.0, 280.0, 150.0, 200.0, 160.0, 190.0, 180.0, 250.0, 285.0, 300.0]

    # 2. Variance components
    σ²a = 20.0
    σ²pe = 12.0
    σ²e = 28.0

    # 3. Incidence matrices and MME setup
    X = [first(incidence(hys)) first(incidence(parity))]
    Z = incidence(cow, nrow(ped))               # breeding values for all 8 animals
    W, _ = incidence(cow)                       # pe only for cows with records (cows 4–8)
    m = mme(
        fy,
        X,
        [
            RandomEffect(Z, ainv(ped), σ²a; name = "animal"),
            RandomEffect(W, I, σ²pe; name = "pe"),
        ];
        σ²e,
    )

    # 4. Solve MME with HYS 1 and 3 constrained to zero
    x = solve_mme(m; zero = [1, 3])             # HYS 1 and 3 set to zero
    b, (a, pe) = solutions(m, x)
    a, pe = vec(a), vec(pe)

    show_table(
        effect = [
            "HYS " .* string.(1:4);
            "parity " .* string.(1:2);
            "animal " .* string.(1:8);
            "pe " .* string.(4:8)
        ],
        solution = x,
    )
    check("HYS and parity", b, [0, 44.065, 0, 0.013, 175.472, 241.893])
    check("animals", a, [10.148, -3.084, -7.063, 13.581, -18.207, -18.387, 9.328, 24.194])
    check("permanent environment", pe, [8.417, -7.146, -17.229, -1.390, 17.347])

    # 5. Daughter Yield Deviation (DYD) of sire 1 (Section 5.2.3):
    #    Adjust phenotypic yields for fixed effects and permanent environment:
    #    yadj = y - Xb - W*pe
    yadj = fy - X * b - W * pe
    nrec = zeros(Int, 8)
    yd = zeros(8)
    for (k, c) in enumerate(cow)
        nrec[c] += 1
        yd[c] += yadj[k]
    end
    yd[4:8] ./= nrec[4:8]
    p = ebv_partition(ped.sire, ped.dam, σ²e / σ²a, nrec, yd, a)

    check("YD of cows 4, 6 and 8", yd[[4, 6, 8]], [23.4005, -38.486, 44.432]; atol = 2e-3)
    check("DYD of sire 1", p.DYD[1], 23.552; atol = 2e-3)

    (; b, a, pe, partition = p)
end
