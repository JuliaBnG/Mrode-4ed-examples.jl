"""
    ex_15_2() -> JointBinaryResult

Example 15.2: Joint analysis of a continuous trait (birth weight, BW) and a binary threshold
trait (calving difficulty, CD) (Mrode & Pocrnic, 2023, Section 15.3; Foulley et al., 1983).

# Model Formulation
Let trait 1 be continuous (birth weight, ``y``) and trait 2 be binary (calving difficulty,
``w \\in \\{0, 1\\}`` governed by an underlying continuous liability ``\\ell``):
```math
\\begin{aligned}
y &= X_1 b_1 + Z_1 u_1 + e_1 \\\\
\\ell &= X_2 b_2 + Z_2 u_2 + e_2
\\end{aligned}
```
with sire transmitting abilities ``u = [u_1';\\ u_2']' \\sim N(0, G_0 \\otimes A)`` and
residuals ``e \\sim N(0, R_0 \\otimes I)``, where:
```math
G_0 = \\begin{bmatrix} \\sigma_{u_1}^2 & \\sigma_{u_{12}} \\\\ \\sigma_{u_{12}} &
    \\sigma_{u_2}^2 \\end{bmatrix}, \\quad
R_0 = \\begin{bmatrix} \\sigma_{e_1}^2 & \\sigma_{e_{12}} \\\\ \\sigma_{e_{12}} & 1
    \\end{bmatrix}
```
(the residual variance of liability is fixed to 1 for identifiability).

Conditioning the liability ``\\ell`` on the observed birth weight ``y`` gives:
```math
\\ell \\mid y = X_2 b_2 + Z_2 u_2 + \\beta (y - X_1 b_1 - Z_1 u_1) + e_{2\\mid 1}
= X_2 \\tau + Z_2 \\nu + \\beta (y - \\bar{y}) + e_{2\\mid 1}
```
where:
- ``\\beta = \\sigma_{e_{12}} / \\sigma_{e_1}^2`` is the residual regression of liability on
  BW.
- ``\\sigma_{e_{2\\mid 1}}^2 = 1 - \\beta^2 \\sigma_{e_1}^2`` is the conditional residual
  variance of liability.
- ``\\tau`` and ``\\nu = u_2 - \\beta u_1`` are the reparameterized fixed and sire effects on
  conditional liability, with BW centred on its mean ``\\bar{y} = 43.02``.

# Key Concepts Illustrated
1. **Calving records and sire summary tables**: All 47 calvings (sire, origin, season, sex,
   BW, and binary CD) are kept in a `DataFrame` (`rec`), and the sire solutions and predicted
   probabilities of difficult calving from both models are collected in a `sires` table.
2. **Joint threshold–linear estimation and probability of difficult calving**: Solving
   iteratively via `joint_binary_quantitative` yields ``(\\hat{b}_1, \\hat{u}_1)`` for BW and
   ``(\\hat{\\tau}, \\hat{\\nu})`` on the conditional liability scale, from which the total
   sire effect on liability is ``\\hat{u}_2 = \\hat{\\nu} + \\hat{\\beta}\\hat{u}_1``. For a
   specific subclass (origin 2, season 1, male calf) and sire ``j``, the probability of a
   difficult calving is
   ``\\Phi\\!\\left(\\hat{\\tau}_{\\text{sub}} + \\hat{\\nu}_j + \\hat{\\beta}(\\hat{b}_{1,\\text{sub}} - \\bar{y})\\right)``.
3. **Bivariate linear model approximation**: Treating the ``0/1`` calving difficulty score as
   a continuous trait gives the standard bivariate MME of [`ex_06_1`](@ref) with sire
   effects, ``[R_0^{-1} \\otimes X'X\\ R_0^{-1} \\otimes X'Z;\\ R_0^{-1} \\otimes Z'X\\
   R_0^{-1} \\otimes Z'Z + G_0^{-1} \\otimes A^{-1}]``, written out with `kron` and verified
   against `mme`.

# Returns
A `JointBinaryResult` with the BW solutions `b1` (fixed) and `u1` (sires), the
liability-scale solutions `τ` (fixed) and `ν` (sires), the residual regression `β`, the
number of iterations `iterations`, and `history`.
"""
function ex_15_2()
    println("Example 15.2: birth weight and calving difficulty")

    # Step 1: Data vectors
    bw = [
        41,
        37.5,
        41.5,
        40,
        43,
        42,
        35,
        46,
        40.5,
        39,
        41.4,
        43,
        34,
        47,
        42,
        44.5,
        49,
        41.6,
        36,
        42.7,
        32.5,
        44.4,
        46,
        47,
        51,
        39,
        44.5,
        40.5,
        43.5,
        42.5,
        48.8,
        38.5,
        52,
        48,
        41,
        50.5,
        43.7,
        51,
        51.6,
        45.3,
        36.5,
        50.5,
        46,
        45,
        36,
        43.5,
        36.5,
    ]
    cd = [
        zeros(11);
        1;
        0;
        1;
        zeros(9);
        1;
        1;
        zeros(5);
        1;
        0;
        0;
        0;
        0;
        ones(5);
        0;
        0;
        1;
        0;
        0;
        0;
        0
    ]
    sex = [
        1,
        1,
        fill(2, 8)...,
        1,
        1,
        2,
        fill(1, 6)...,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        2,
        1,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
    ]
    origin = [
        fill(1, 7);
        fill(2, 3);
        fill(1, 5);
        2;
        2;
        1;
        fill(2, 5);
        1;
        1;
        1;
        2;
        fill(1, 7);
        2;
        2;
        2;
        2;
        fill(1, 7);
        2;
        2
    ]
    season = [
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        1,
    ]
    sire = [fill(1, 10); fill(2, 7); fill(3, 6); fill(4, 4); fill(5, 11); fill(6, 9)]
    #    one row per calving (sex 1 = male, 2 = female)
    rec = DataFrame(; sire, origin, season, sex, bw = Float64.(bw), cd)
    println(first(rec, 6), "\n  … ", nrow(rec), " calvings")

    # Step 2: Sire – maternal grandsire pedigree relationship matrix
    Ai = Matrix(ainv_smgs(DataFrame(sire = [0, 0, 1, 2, 3, 2], mgs = [0, 0, 0, 1, 2, 3])))
    check("A⁻¹ (sire–MGS), first row", Ai[1, :], [1.424, 0.182, -0.667, -0.364, 0, 0])

    # Step 3: Design matrices and genetic covariance
    X = [
        first(incidence(rec.origin)) first(incidence(rec.season)) first(incidence(rec.sex))
    ]
    Z, _ = incidence(rec.sire)
    G = [0.7178 0.1131; 0.1131 0.0466]

    # Step 4: Joint threshold-linear estimation
    r = joint_binary_quantitative(
        rec.bw,
        rec.cd,
        X,
        Z,
        X,
        Z,
        Ai,
        G,
        20.0,
        0.459;
        ycentre = 43.02,
        zero1 = [4, 6],
        zero2 = [4, 6],
    )
    check("residual regression β", r.β, 0.1155)

    names = [
        "origin 1",
        "origin 2",
        "season 1",
        "season 2",
        "male",
        "female",
        "sire " .* string.(1:6)...,
    ]
    show_table(effect = names[1:6], BW = r.b1, CD = r.τ; digits = 4)

    check(
        "iteration 0: BW",
        r.history[1:12, 1],
        [
            41.6177,
            42.2069,
            -1.2359,
            0,
            3.1728,
            0,
            -0.3497,
            0.1201,
            -0.2852,
            0.2022,
            0.2994,
            0.1794,
        ];
        atol = 1e-3,
    )
    check(
        "iteration 0: CD",
        r.history[13:24, 1],
        [
            0.1343,
            0.0851,
            -0.0317,
            0,
            0.2437,
            0,
            -0.0450,
            0.0237,
            -0.0363,
            0.0289,
            0.0188,
            0.0272,
        ];
        atol = 1e-3,
    )
    check(
        "converged: BW",
        [r.b1; r.u1],
        [
            41.6192,
            42.2109,
            -1.2344,
            0,
            3.1690,
            0,
            -0.3592,
            0.1303,
            -0.2948,
            0.2126,
            0.2969,
            0.1815,
        ];
        atol = 1e-3,
    )
    check(
        "converged: CD",
        [r.τ; r.ν],
        [
            -1.3936,
            -1.7457,
            0.1404,
            0,
            0.8401,
            0,
            -0.0573,
            0.0372,
            -0.0485,
            0.0411,
            0.0152,
            0.0309,
        ];
        atol = 1e-3,
    )
    println("  converged after ", r.iterations, " iterations")

    # Step 5: Probability of difficult calving for profile:
    # heifer of origin 2, season 1, male calf
    V = [
        cdf(
            Normal(),
            r.τ[2] + r.τ[3] + r.τ[5] + r.ν[j] + r.β * (r.b1[2] + r.b1[3] + r.b1[5] - 43.02),
        ) for j = 1:6
    ]
    check(
        "P(difficult | origin 2, season 1, male)",
        V,
        [0.245, 0.275, 0.247, 0.276, 0.268, 0.273];
        atol = 2e-3,
    )

    # Step 6: Bivariate linear model approximation (Section 15.3). Both traits are
    #    recorded on every calving, so the MME are those of Example 6.1, here with
    #    sire effects: [R₀⁻¹⊗X'X R₀⁻¹⊗X'Z; R₀⁻¹⊗Z'X R₀⁻¹⊗Z'Z + G₀⁻¹⊗A⁻¹]
    R0 = [20 2.089; 2.089 1.036]
    Y = [rec.bw rec.cd]
    keepX = setdiff(1:6, [4, 6])
    Xk = X[:, keepX]
    R0i = inv(R0)
    #! format: off
    lhs = [
        kron(R0i, Xk'Xk)  kron(R0i, Xk'Z)
        kron(R0i, Z'Xk)   kron(R0i, Z'Z) + kron(inv(G), Ai)
    ]
    #! format: on
    rhs = [kron(R0i, Xk'); kron(R0i, Z')] * vec(Y)
    y, obs = mt_stack(Y)
    m = mme(
        y,
        reduce(hcat, mt_blocks(obs, [Xk, Xk])),
        [RandomEffect(mt_blocks(obs, [Z, Z]), Ai, G)];
        Rinv = mt_rinv(R0, obs),
    )
    check("linear model: mme() LHS = Kronecker form", m.lhs, lhs; atol = 1e-10)
    x = lhs \ rhs
    b, u = x[m.blocks[1]], reshape(x[m.blocks[2]], :, 2)
    check(
        "linear model: BW",
        [b[1:4]; u[:, 1]],
        [
            41.6175,
            42.2022,
            -1.2387,
            3.1845,
            -0.3268,
            0.1171,
            -0.2641,
            0.1886,
            0.2688,
            0.1690,
        ];
        atol = 1e-3,
    )
    check(
        "linear model: CD",
        [b[5:8]; u[:, 2]],
        [0.1349, 0.0876, -0.0311, 0.2410, -0.0527, 0.0285, -0.0427, 0.0350, 0.0195, 0.0323];
        atol = 1e-3,
    )

    sires =
        DataFrame(sire = 1:6, u1 = r.u1, ν = r.ν, u2 = r.ν + r.β * r.u1, P_difficult = V)
    sires.lin_bw, sires.lin_cd = u[:, 1], u[:, 2]
    println("\nSires: BW (u1) and liability (ν, u2 = ν + βu1) solutions, probability of")
    println("difficult calving (origin 2, season 1, male), and the linear-model solutions:")
    show_table(sires; digits = 4)

    r
end
