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
1. **Handling Missing Multi-Trait Records**: `mt_stack`, `mt_blocks`, and `mt_rinv`
   automatically invert the appropriate submatrices according to observed patterns.
2. **Information Borrowing Across Correlated Traits**:
   Even without a record on PWG, calves 4 and 9 obtain PWG EBVs from their own
   WWG records (via genetic correlation ``r_g = 0.636``) and their relatives' records.
3. **Comparison with Univariate Analysis**:
   A univariate analysis of PWG cannot use the WWG records of calves 4 and 9.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed sex effect solutions for WWG and PWG.
- `a`: Matrix of estimated breeding values for all 9 animals.
"""
function ex_06_2()
    println("Example 6.2: bivariate analysis with missing PWG records")

    # 1. Pedigree and multi-trait records with missing entries
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3, 7], dam = [0, 0, 0, 0, 2, 2, 5, 6, 0])
    calf = 4:9
    sex = ["M", "F", "F", "M", "M", "F"]
    Y = [4.5 missing; 2.9 5.0; 3.9 6.8; 3.5 6.0; 5.0 7.5; 4.0 missing]

    # 2. Covariance matrices
    G0 = [20.0 18.0; 18.0 40.0]
    R0 = [40.0 11.0; 11.0 30.0]

    # 3. Multi-trait incidence matrices and MME setup
    Ai = ainv(ped)
    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(calf, 9)
    y, obs = mt_stack(Y)
    Ri = mt_rinv(R0, obs)       # inverses of R_o and R_m by missing pattern
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [X, X])),
        [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G0)];
        Rinv = Ri,
    )

    # 4. Solve multi-trait MME
    b, (a,) = solutions(m, solve_mme(m))
    check("X₁'R¹¹X₁", Matrix(m.lhs[1:2, 1:2]), [0.081 0; 0 0.081])

    # 5. Univariate analyses using only observed records of each trait
    uni = map(1:2) do t
        r = findall(obs[:, t])
        mu = mme(
            Float64.(Y[r, t]),
            X[r, :],
            [RandomEffect(Z[r, :], Ai, G0[t, t])];
            σ²e = R0[t, t],
        )
        solve_mme(mu)
    end

    show_table(
        effect = ["male", "female", string.(1:9)...],
        WWG = [b[1:2]; a[:, 1]],
        PWG = [b[3:4]; a[:, 2]],
        WWG_uni = uni[1],
        PWG_uni = uni[2],
    )
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

    (; b, a)
end
