"""
    ex_06_4() -> NamedTuple

Example 6.4: Multivariate model with zero residual covariance (sex-limited traits in
relatives) (Mrode & Pocrnic, 2023, Section 6.5).

# Model Formulation
When traits are sex-limited (e.g. yearling weight recorded only on males and fat yield
recorded only on females), no animal can have records on both traits simultaneously.
The residual covariance between the traits is therefore not estimable and is set
to zero:
```math
R_0 = \\begin{bmatrix} \\sigma_{e1}^2 & 0 \\\\ 0 & \\sigma_{e2}^2 \\end{bmatrix} =
\\begin{bmatrix} 77.0 & 0 \\\\ 0 & 70.0 \\end{bmatrix}
```
while the genetic covariance is non-zero:
```math
G_0 = \\begin{bmatrix} 43.0 & 18.0 \\\\ 18.0 & 30.0 \\end{bmatrix}
```

# Key Concepts Illustrated
1. **Sex-Limited Traits and Correlated Relatives**: Information flows across traits
   strictly through genetic relationships in ``A^{-1}`` and genetic covariance ``G_0``.
2. **Unequal Fixed Effect Levels**: Yearling weight has only 2 HYS levels, while fat yield
   has 3 HYS levels.

# Note on Text Discrepancy
The book prints animal 6 as ``(-5.012, -2.098)``, identical to animal 5.
However, dam 6 has a son (animal 11) with a high yearling weight record (300 kg in HYS 2),
giving her positive evaluations ``(5.012, 2.098)``, as verified by the MME solutions.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed HYS effects (levels 1-2 for yearling weight, 1-3 for fat yield).
- `a`: Matrix of estimated breeding values for all 17 animals across both traits.
"""
function ex_06_4()
    println("Example 6.4: different traits recorded on relatives")

    # 1. Pedigree and sex-limited records (calves 9–17)
    sire = [0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 3, 1, 3, 2, 2, 3]
    dam = [0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 7, 8, 0, 13, 15]
    ped = DataFrame(; sire, dam)
    calf = 9:17
    hys = [1, 2, 2, 1, 1, 2, 3, 2, 3]

    # Yearling weight (males 9-12), Fat yield (females 13-17)
    Y = [
        375.0 missing;
        250.0 missing;
        300.0 missing;
        450.0 missing;
        missing 200.0;
        missing 160.0;
        missing 150.0;
        missing 250.0;
        missing 175.0
    ]

    # 2. Covariance matrices (zero environmental covariance)
    G0 = [43.0 18.0; 18.0 30.0]
    R0 = [77.0 0.0; 0.0 70.0]

    # 3. Stack records and setup design matrices
    y, obs = mt_stack(Y)
    X, _ = incidence(hys)
    Z = incidence(calf, 17)
    Xb = mt_blocks(obs, [X, X])
    Xb[1] = Xb[1][:, 1:2]                   # no HYS 3 for yearling weight
    m = mme(
        y,
        reduce(hcat, Xb),
        [RandomEffect(mt_blocks(obs, [Z, Z]), ainv(ped), G0)];
        Rinv = mt_rinv(R0, obs),
    )

    # 4. Solve MME
    b, (a,) = solutions(m, solve_mme(m))

    show_table(animal = 1:17, weight = a[:, 1], fat = a[:, 2])
    check("HYS, yearling weight", b[1:2], [412.26, 276.21]; atol = 6e-3)
    check("HYS, fat", b[3:5], [194.03, 204.77, 161.66]; atol = 5e-3)

    # Note on book typo:
    # The book prints animal 6 as (-5.012, -2.098), a copy of animal 5; dam 6
    # has a son (11) with a high record, so her values are positive (+5.012, +2.098)
    check(
        "animals",
        a,
        [
            -3.365 1.258;
            -1.489 3.774;
            4.237 -1.687;
            -6.940 -1.572;
            -5.012 -2.098;
            5.012 2.098;
            2.137 3.561;
            -4.274 -7.123;
            -12.162 -3.091;
            -8.263 -1.260;
            5.836 3.776;
            12.632 3.558;
            1.523 5.971;
            -4.292 -11.527;
            -1.870 0.011;
            4.290 11.995;
            2.684 1.663
        ],
    )

    (; b, a)
end
