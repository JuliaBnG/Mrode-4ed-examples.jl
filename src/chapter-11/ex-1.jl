"""
    ex_11_1() -> Vector{NamedTuple}

Example 11.1: Fitting SNPs as fixed effects alongside a random polygenic animal effect
(Mrode & Pocrnic, 2023, Sections 11.2–11.3).

# Model Formulation
A hybrid model fitting major QTL / candidate SNPs as fixed regressions while modeling
background polygenic variance with a pedigree relationship matrix:
```math
y = Xb + Z_m g + W u + e
```
where:
-  ``y`` is the vector of Daughter Yield Deviations (DYD) of the 8 reference bulls (bulls 13
  to 20).
- ``b`` is the overall fixed mean.
-  ``g`` is the vector of fixed regression effects for the first 3 SNPs (centered genotypes
  ``Z_m = M - 2p``).
-  ``u`` is the random polygenic animal effect for all 26 bulls, with
  ``\\mathrm{var}(u) = A \\sigma_a^2``.
- Both unweighted analysis (``\\mathrm{var}(e) = I \\sigma_e^2``) and weighted analysis
  (``\\mathrm{var}(e) = \\mathrm{diag}(\\sigma_e^2 / \\mathrm{EDC}_i)``) are evaluated.
- Direct Genomic Value (DGV) is computed as ``\\mathrm{DGV} = Z_m \\hat{g}``, and
  Genomic Estimated Breeding Value (GEBV) as ``\\mathrm{GEBV} = \\mathrm{DGV} + \\hat{u}``.

# Key Concepts Illustrated
1. **Tables**: the pedigree, and the genotyped bulls with their DYD and EDC;
   DGV, polygenic effects and GEBV of both analyses are added as columns of
   the bull table. The genotypes stay a matrix, `M`.
2. **One MME for both analyses**: the general form
   ```math
   \\begin{bmatrix}
   X'R^{-1}X & X'R^{-1}W \\\\
   W'R^{-1}X & W'R^{-1}W + A^{-1}/\\sigma_a^2
   \\end{bmatrix},
   \\quad X = [\\mathbf{1}\\ Z_m],
   ```
   with ``R^{-1} = I/\\sigma_e^2`` (unweighted) or
   ``R^{-1} = \\mathrm{diag}(\\mathrm{EDC}_i/\\sigma_e^2)`` (weighted), written out and
   compared with `mme`.

# Returns
A `Vector{NamedTuple}` with unweighted and weighted results:
`[(; b, u, dgv), (; b, u, dgv)]`.
"""
function ex_11_1()
    println("Example 11.1: three SNPs as fixed effects plus polygenic effects")

    # 1. Pedigree (one row per animal), bulls (one row per genotyped bull, in the
    #    row order of the genotype matrix M) and variance components
    (; ped, bulls, M, σ²a, σ²e) = data_11()
    println(bulls)

    # 2. Center genotypes: Z = M - 2p
    Z, p, _ = center_genotypes(M)
    check(
        "allele frequencies",
        p,
        [0.321, 0.179, 0.357, 0.357, 0.143, 0.607, 0.071, 0.964, 0.571, 0.393],
    )

    # 3. Design matrices for the reference bulls: mean and the first 3 SNPs
    #    (fixed), and the polygenic effect of all 26 animals
    isref = bulls.set .== "reference"
    ref = bulls[isref, :]
    Z3 = Z[:, 1:3]
    y = ref.dyd
    X = [ones(nrow(ref)) Z3[isref, :]]
    W = incidence(ref.id, nrow(ped))
    Ai = ainv(ped)

    # 4. MME in the general form [X'R⁻¹X X'R⁻¹W; W'R⁻¹X W'R⁻¹W + A⁻¹/σ²a] for
    #    the unweighted analysis, R⁻¹ = I/σ²e, and the weighted one,
    #    R⁻¹ = diag(EDC/σ²e): each DYD is weighted by its daughter contribution
    fits = [("unweighted", fill(1 / σ²e, nrow(ref))), ("weighted", ref.edc ./ σ²e)]
    res = map(fits) do (lab, rinv)
        Ri = Diagonal(rinv)
        #! format: off
        lhs = [
            X'Ri*X  X'Ri*W
            W'Ri*X  W'Ri*W + Ai / σ²a
        ]
        #! format: on
        rhs = [X'Ri*y; W'Ri*y]
        m = mme(y, X, [RandomEffect(W, Ai, σ²a)]; Rinv = rinv)
        check("$lab: mme(; Rinv) LHS = general form", m.lhs, lhs; atol = 1e-12)
        x = lhs \ rhs
        b = x[m.blocks[1]]
        u = x[m.blocks[2]]
        (; b, u, dgv = Z3 * b[2:4])
    end
    #    With R = Iσ²e, the λ form of mme(; σ²e) gives the same solutions
    m0 = mme(y, X, [RandomEffect(W, Ai, σ²a)]; σ²e)
    check(
        "unweighted: λ form = general form",
        solve_mme(m0),
        [res[1].b; res[1].u];
        atol = 1e-10,
    )

    # 5. Results per bull: DGV = Z₃ĝ, polygenic û, GEBV = DGV + û
    for ((lab, _), r) in zip(fits, res)
        bulls[!, "dgv_$lab"] = r.dgv
        bulls[!, "u_$lab"] = r.u[bulls.id]
        bulls[!, "gebv_$lab"] = r.dgv + r.u[bulls.id]
    end
    show_table(bulls)

    uw, wt = res
    check("unweighted: mean and SNP effects", uw.b, [9.895, 0.607, -4.080, 1.934])
    check("weighted: mean and SNP effects", wt.b, [9.196, 1.158, -3.956, 2.527])
    check(
        "unweighted: DGV",
        uw.dgv,
        [
            2.834,
            0.293,
            0.081,
            3.554,
            -2.460,
            -3.787,
            1.620,
            -2.460,
            0.900,
            -0.314,
            -2.460,
            0.293,
            -0.314,
            2.227,
        ],
    )
    check(
        "unweighted: polygenic",
        uw.u[bulls.id],
        [
            -0.299,
            0.256,
            0.142,
            0.254,
            -0.085,
            0.271,
            -0.092,
            -0.181,
            0,
            0.128,
            0.128,
            0.128,
            0.128,
            0.128,
        ],
    )
    check(
        "weighted: DGV",
        wt.dgv,
        [
            3.708,
            0.022,
            1.120,
            3.918,
            -2.565,
            -3.934,
            1.391,
            -2.565,
            1.181,
            -1.136,
            -2.565,
            0.022,
            -1.136,
            2.549,
        ];
        atol = 3e-3,
    )  # large EDC weights magnify the book's rounding (α = 6.952)
    check(
        "weighted: polygenic",
        wt.u[bulls.id],
        [
            -3.821,
            4.116,
            2.248,
            2.158,
            -0.450,
            2.402,
            -0.212,
            -1.542,
            0,
            2.058,
            2.058,
            2.058,
            2.058,
            2.058,
        ];
        atol = 3e-3,
    )
    check("GEBV of bull 13 (unweighted)", bulls.gebv_unweighted[1], 2.535)

    res
end
