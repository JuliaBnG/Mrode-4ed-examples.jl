"""
    ex_06_2() -> NamedTuple

Example 6.2: Multivariate animal model with equal design matrices and missing records
(Mrode & Pocrnic, 2023, Section 6.3).

# Model Formulation
When some animals lack records for one or more traits (e.g. calves 4 and 9 have WWG but
missing PWG), the residual covariance structure cannot be represented by a simple Kronecker
product ``R_0 \\otimes I``. Instead:
```math
R^{-1} = \\bigoplus_{i=1}^n R_i^{-1}
```
where ``R_i`` is the submatrix of ``R_0`` corresponding to the traits observed on individual
``i``.

# Key Concepts Illustrated
1. **One table for the data**: pedigree, sex and both traits of all 9 animals
   in one `DataFrame`, with `missing` for the PWG of calves 4 and 9;
   multi-trait and univariate EBV are added as columns.
2. **Handling Missing Multi-Trait Records**: the MME of Example 6.1 still
   hold if each missing record is set to 0 and ``R^{-1}`` is padded, i.e. a
   calf with observed traits ``S`` contributes ``(R_0[S,S])^{-1}`` on those
   traits and zeros elsewhere:
   ```math
   \\begin{bmatrix}
   (I_2 \\otimes X)' R^{-1} (I_2 \\otimes X) &
       (I_2 \\otimes X)' R^{-1} (I_2 \\otimes Z) \\\\
   (I_2 \\otimes Z)' R^{-1} (I_2 \\otimes X) &
       (I_2 \\otimes Z)' R^{-1} (I_2 \\otimes Z) + G_0^{-1} \\otimes A^{-1}
   \\end{bmatrix}.
   ```
   This form is written out explicitly and compared with `mt_stack`,
   `mt_blocks`, and `mt_rinv`, which drop the missing records instead and
   invert the submatrices of ``R_0`` once per missing pattern.
3. **Information Borrowing Across Correlated Traits**:
   Even without a record on PWG, calves 4 and 9 obtain PWG EBVs from their own
   WWG records (via genetic correlation ``r_g = 0.636``) and their relatives' records.
4. **Comparison with Univariate Analysis**:
   A univariate analysis of PWG cannot use the WWG records of calves 4 and 9.

# Returns
A `NamedTuple` with fields:
- `ped`: The data with columns `ebv_wwg`, `ebv_pwg`, `uni_wwg` and `uni_pwg` added.
- `b`: Fixed sex effect solutions for WWG and PWG.
- `a`: Matrix of estimated breeding values for all 9 animals.
"""
function ex_06_2()
    println("Example 6.2: bivariate analysis with missing PWG records")

    # 1. Pedigree and records of both traits, one row per animal
    #    (IDs are row numbers; founders 1–3 have no records, calves 4 and 9 no PWG)
    no3 = fill(missing, 3)
    ped = DataFrame(
        calf = 1:9,
        sex = [no3; ["M", "F", "F", "M", "M", "F"]],
        sire = [0, 0, 0, 1, 3, 1, 4, 3, 7],
        dam = [0, 0, 0, 0, 2, 2, 5, 6, 0],
        wwg = [no3; [4.5, 2.9, 3.9, 3.5, 5.0, 4.0]],
        pwg = [no3; [missing, 5.0, 6.8, 6.0, 7.5, missing]],
    )
    @assert ped.calf == 1:nrow(ped)
    rec = dropmissing(ped, :wwg)                # calves with at least one record
    Y = Matrix(rec[:, [:wwg, :pwg]])            # col 1: WWG, col 2: PWG, with missing
    println(ped)

    # 2. Covariance matrices
    G0 = [20.0 18.0; 18.0 40.0]
    R0 = [40.0 11.0; 11.0 30.0]

    # 3. Incidence matrices, the same for both traits
    Ai = ainv(ped)
    X, _ = incidence(rec.sex; levels = ["M", "F"])
    Z = incidence(rec.calf, nrow(ped))

    # 4a. MME written out as in Example 6.1, but with each missing record set to 0
    #     and R⁻¹ padded: a calf with observed traits S contributes (R₀[S,S])⁻¹
    #     on those traits and zeros on the missing ones.
    obs = .!ismissing.(Y)
    Rs = map(eachrow(obs)) do o
        Rk = zeros(2, 2)
        Rk[o, o] = inv(R0[o, o])
        Rk
    end
    println("\nR⁻¹ of a calf with both records:")
    show(stdout, "text/plain", round.(Rs[2]; digits = 4))
    println("\nR⁻¹ of a calf with WWG only:")
    show(stdout, "text/plain", round.(Rs[1]; digits = 4))
    println()
    #     Residual precision of the trait-major stacked records [WWG; PWG]
    D(j, k) = Diagonal(getindex.(Rs, j, k))
    #! format: off
    Ri = [
        D(1, 1)  D(1, 2)
        D(2, 1)  D(2, 2)
    ]
    #! format: on
    X2 = kron(I(2), X)                          # [X 0; 0 X]
    Z2 = kron(I(2), Z)                          # [Z 0; 0 Z]
    y0 = coalesce.(vec(Y), 0.0)                 # missing → 0; their R⁻¹ rows are 0
    #! format: off
    lhs = [
        X2'Ri*X2  X2'Ri*Z2
        Z2'Ri*X2  Z2'Ri*Z2 + kron(inv(G0), Ai)
    ]
    #! format: on
    rhs = [X2'Ri*y0; Z2'Ri*y0]

    # 4b. The same equations with mt_stack, mt_blocks and mt_rinv, where the
    #     missing records are dropped instead of padded
    y, _ = mt_stack(Y)                          # observed records only, trait-major
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [X, X])),
        [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0; name = "animal")];
        Rinv = mt_rinv(R0, obs),
    )
    check("mme() LHS = padded form", m.lhs, lhs; atol = 1e-12)
    check("mme() RHS = padded form", m.rhs, rhs; atol = 1e-12)
    check("X₁'R¹¹X₁", Matrix(lhs[1:2, 1:2]), [0.081 0; 0 0.081])

    # 5. Solve multi-trait MME
    x = lhs \ rhs
    b = x[m.blocks[1]]
    a = reshape(x[m.blocks[2]], :, 2)           # rows: animals, cols: WWG and PWG

    # 6. Univariate analyses using only the observed records of each trait
    uni = map(enumerate([:wwg, :pwg])) do (t, col)
        r = dropmissing(ped, col)
        Xt, _ = incidence(r.sex; levels = ["M", "F"])
        Zt = incidence(r.calf, nrow(ped))
        solve_mme(mme(r[!, col], Xt, [RandomEffect(Zt, Ai, G0[t, t])]; σ²e = R0[t, t]))
    end

    println("\nSex effects:")
    show_table(
        sex = ["M", "F"],
        WWG = b[1:2],
        PWG = b[3:4],
        WWG_uni = uni[1][1:2],
        PWG_uni = uni[2][1:2],
    )
    ped.ebv_wwg, ped.ebv_pwg = a[:, 1], a[:, 2]
    ped.uni_wwg, ped.uni_pwg = uni[1][3:end], uni[2][3:end]
    check("sex (WWG, PWG)", b, [4.367, 3.657, 6.834, 6.007])
    check(
        "animals",
        a,
        [
            0.130 0.266;
            -0.084 -0.075;
            -0.098 -0.194;
            0.007 0.016;
            -0.343 -0.555;
            0.192 0.440;
            -0.308 -0.483;
            0.201 0.349;
            -0.018 -0.119
        ],
    )
    check(
        "univariate WWG",
        uni[1],
        [4.364, 3.648, 0.077, -0.081, -0.058, 0.003, -0.250, 0.098, -0.237, 0.143, 0.010],
    )
    check(
        "univariate PWG",
        uni[2],
        [6.784, 5.873, 0.273, 0.000, -0.165, -0.025, -0.463, 0.517, -0.460, 0.392, -0.230],
    )

    println("\nData with multi-trait and univariate EBV:")
    show_table(ped)

    (; ped, b, a)
end
