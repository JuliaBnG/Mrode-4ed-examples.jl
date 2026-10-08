"""
    ex_15_1() -> ThresholdResult

Example 15.1: Threshold (probit) sire model for calving ease in three ordered categories
(Mrode & Pocrnic, 2023, Sections 15.1–15.2; Gianola & Foulley, 1983; Harville & Mee, 1984).

# Model Formulation and Theory
Calving ease is scored in 3 ordered categories (1 = normal, 2 = slight difficulty, 3 =
extreme difficulty) governed by an underlying unobservable continuous liability ``\\ell``:
```math
\\ell = Xb + Z s + e, \\quad e \\sim N(0, 1)
```
with fixed thresholds ``t_1 < t_2`` (setting ``t_0 = -\\infty, t_3 = +\\infty``) such that
an observation falls in category ``k`` if ``t_{k-1} < \\ell \\le t_k``. The probability of
observing category ``k`` given linear predictor ``\\eta = x' b + z' s`` is:
```math
P(Y = k \\mid \\eta) = \\Phi(t_k - \\eta) - \\Phi(t_{k-1} - \\eta)
```
where:
- ``\\Phi(\\cdot)`` is the standard normal cumulative distribution function (probit link).
-  The residual variance on the liability scale is set to ``\\sigma_e^2 = 1.0`` for
  identifiability.
-  Sire transmitting ability variance on the liability scale is ``\\sigma_s^2 = 1/19``
  (variance ratio ``\\alpha = 19``).
-  Thresholds ``t`` and location parameters ``\\theta = [b; s]`` are estimated jointly by
  Fisher scoring (iteratively reweighted generalized linear mixed model via
  `threshold_model`).
- Category probabilities by sire subclass are computed with `category_probabilities`.

# Returns
A `ThresholdResult` object containing converged thresholds `t`, fixed effects `b`, sire
effects `u`, and iteration history.
"""
function ex_15_1()
    println("Example 15.1: threshold model for calving ease")

    # 1. Calving ease category counts N (20 herd-sex-sire subclasses × 3 categories)
    herd = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    sex = [
        "M",
        "F",
        "M",
        "F",
        "M",
        "F",
        "M",
        "F",
        "M",
        "F",
        "M",
        "M",
        "F",
        "M",
        "F",
        "M",
        "M",
        "F",
        "F",
        "M",
    ]
    sire = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
    N = [
        1 0 0;
        1 0 0;
        1 0 0;
        0 1 0;
        1 0 1;
        3 0 0;
        1 1 0;
        0 1 0;
        1 0 0;
        2 0 0;
        1 0 0;
        0 0 1;
        1 0 1;
        1 0 0;
        0 1 0;
        0 0 1;
        0 1 0;
        1 0 0;
        2 0 0;
        2 0 0
    ]

    # 2. Sire pedigree and relationship matrix
    sped = DataFrame(sire = [0, 0, 1, 3], dam = [0, 0, 0, 0])
    Ai = Matrix(ainv(sped))

    # 3. Design matrices and threshold model fitting (Gianola & Foulley, 1983)
    X = [first(incidence(herd)) first(incidence(sex; levels = ["M", "F"]))]
    Z, _ = incidence(sire)
    r = threshold_model(N, X, Z, Ai, 19; t0 = [0.468, 1.080], zero = [1, 3])

    check(
        "after iteration 1: t, b, u",
        r.history[:, 1],
        [
            0.441008,
            1.044792,
            0,
            0.286869,
            0,
            -0.358323,
            -0.041528,
            0.057853,
            0.039850,
            -0.065178,
        ];
        atol = 1e-5,
    )
    check(
        "after iteration 2",
        r.history[:, 2],
        [0.4375, 1.0661, 0, 0.2763, 0, -0.3577, -0.0431, 0.0586, 0.0410, -0.0653];
        atol = 1e-4,
    )
    println("  converged after ", r.iterations, " iterations")

    # 4. Standard errors from the inverted Fisher information / coefficient matrix
    keep = setdiff(1:10, [3, 5])
    C = zeros(10, 10)
    C[keep, keep] = inv(r.lhs[keep, keep])
    se = sqrt.(diag(C))

    show_table(
        effect = [
            "t1",
            "t2",
            "herd 1",
            "herd 2",
            "male",
            "female",
            "sire " .* string.(1:4)...,
        ],
        solution = [r.t; r.b; r.u],
        se = se;
        digits = 4,
    )
    check(
        "final solutions",
        [r.t; r.b; r.u],
        [0.4378, 1.0675, 0, 0.2774, 0, -0.3590, -0.0434, 0.0592, 0.0412, -0.0660];
        atol = 1e-4,
    )
    check(
        "standard errors",
        se[[1, 2, 4, 6, 7, 8, 9, 10]],
        [0.44, 0.47, 0.49, 0.48, 0.22, 0.21, 0.22, 0.22];
        atol = 0.005,
    )

    # 5. Sire-specific category probabilities:
    #    (a) For female calves in herd–year 1:
    η = r.b[1] + r.b[4] .+ r.u
    P = category_probabilities(r.t, η)
    check(
        "P(category | herd 1, female, sire)",
        P,
        [0.800 0.129 0.071; 0.770 0.145 0.086; 0.775 0.142 0.083; 0.803 0.129 0.068];
        atol = 3e-3,
    )

    #    (b) Marginal probabilities averaged over the four herd–year × sex subclasses:
    Pbar =
        sum(category_probabilities(r.t, r.b[h] + r.b[2+s] .+ r.u) for h = 1:2, s = 1:2) ./ 4
    check(
        "P(category | sire), all subclasses",
        Pbar,
        [0.695 0.175 0.131; 0.659 0.188 0.153; 0.665 0.186 0.149; 0.702 0.172 0.126];
        atol = 2e-3,
    )

    # 6. Comparison with linear sire model on raw category scores (1, 2, 3)
    rec = [(j, k) for j in axes(N, 1) for k = 1:3 for _ = 1:N[j, k]]
    j, y = first.(rec), Float64.(last.(rec))
    m = mme(y, X[j, :], [RandomEffect(Z[j, :], Ai, 1 / 19)]; σ²e = 1.0)
    x = solve_mme(m; zero = [1])
    show_table(
        effect = ["herd 2", "male", "female", "sire " .* string.(1:4)...],
        linear = x[2:8];
        digits = 4,
    )

    r
end
