"""
    ex_14_1() -> NamedTuple

Examples 14.1 & 14.2: Multibreed animal models with breed-specific partial relationship
matrices (Mrode & Pocrnic, 2023, Sections 14.2–14.4; García-Cortés & Toro, 2006; Strandén &
Mäntysaari, 2013).

# Model Formulation
In crossbred and admixed populations with breeds 1 and 2:
1.  **Breed-Specific Partial Relationship Matrices** (Example 14.1; García-Cortés & Toro,
   2006): The total additive genetic variance is partitioned into breed-origin components:
   ```math
   a = a^{(1)} + a^{(2)} + a^{(12)}
   ```
   with ``\\mathrm{var}(a^{(1)}) = A_1 \\sigma_1^2``,
   ``\\mathrm{var}(a^{(2)}) = A_2 \\sigma_2^2``, and segregation variance
   ``\\mathrm{var}(a^{(12)}) = A_{12} \\sigma_{12}^2``. ``A_1, A_2`` are constructed from
   breed compositions ``F`` via `partial_nrm`, and ``A_{12}`` from segregation coefficients
   ``h_{12}``.
2.  **Equivalent Combined Model**: Solves directly for total merit with overall covariance
   ``G = \\sigma_1^2 A_1 + \\sigma_2^2 A_2 + \\sigma_{12}^2 A_{12}``, proving
   ``\\hat{a} = \\hat{a}^{(1)} + \\hat{a}^{(2)} + \\hat{a}^{(12)}``.
3. **Random-Regression Approximation** (Example 14.2; Strandén & Mäntysaari, 2013):
   Approximates the partial relationship model using the standard pedigree ``A`` with
   covariables ``\\sqrt{f_1}, \\sqrt{f_2}, \\sqrt{h_{12}}``.

# Key Concepts Illustrated
1. **One table for all animals**: Pedigree, herd, phenotype, breed fractions (`f1`, `f2`),
   segregation coefficients (`h12`), and the genetic predictions from all three models are
   kept side by side as columns of `animals`.
2. **Writing out the MME of all three models as ``T'T + \\Sigma``**:
   - **Model 1 (partial relationships)**: ``T = [X\\ Z_1\\ Z_2\\ Z_{12}]``, where ``Z_k``
     picks the animals with ``f_k > 0`` (or ``h_{12} > 0``) and ``\\Sigma`` holds
     ``\\sigma_e^2/\\sigma_k^2`` times the generalized inverses of ``A_1, A_2, A_{12}``
     (animals with zero contribution in a partial matrix have zero rows/columns and are
     constrained to zero).
   - **Model 2 (combined model)**: With one record per animal (``Z = I``), the MME reduce to
     ``[X'X\\ X';\\ X\\ I + \\sigma_e^2 G^{-1}]`` and give ``\\hat{a} = \\hat{a}^{(1)} +
     \\hat{a}^{(2)} + \\hat{a}^{(12)}``.
   - **Model 3 (random-regression approximation, Example 14.2)**: ``T`` uses the diagonal
     covariables ``\\mathrm{Diag}(\\sqrt{f_1}), \\mathrm{Diag}(\\sqrt{f_2}),
     \\mathrm{Diag}(\\sqrt{h_{12}})``, and ``\\Sigma`` uses the full-pedigree ``A``
     restricted to the animals carrying each component.
   All three explicit systems are checked against `mme`.

# Note on Text Discrepancy
In Example 14.2, Table 14.3 of the book lists the weighted breed contributions
``\\sqrt{f_p}\\,\\hat{u}_p`` and ``\\sqrt{h_{12}}\\,\\hat{u}_{12}`` (which sum directly to the
total breeding value ``\\hat{a}^*``) rather than the raw random-regression coefficients
``\\hat{u}_p``.

# Returns
A `NamedTuple` with fields:
- `b`: Fixed effect solutions (herd and breed) from the partial relationship model.
- `u`: Breed-specific solutions (breed 1, breed 2, segregation).
- `a`: Total breeding values from the combined model.
- `br`: Fixed effect solutions from the random-regression model.
- `ur`: Random-regression coefficients.
- `bv`: Total breeding values from the random-regression model.
- `animals`: The animal table with `h12`, `u1`, `u2`, `u12`, `a`, and `a_rr` added.
"""
function ex_14_1()
    println("Example 14.1: breed-specific and combined multibreed models")

    # 1. Load pedigree, breed composition, and variance components
    (; ped, animals, F, herd, y, σ², σ²e) = data_14_1()
    n = nrow(animals)

    # 2. Compute segregation coefficients h₁₂ and partial relationship matrices A₁, A₂, A₁₂
    h12 = segregation_coefficients(ped, F, 1, 2)
    animals.h12 = h12
    check(
        "f₁ and h₁₂",
        [F[:, 1] h12],
        [1 0; 1 0; 0 0; 0 0; 1 0; 0.5 0; 0 0; 0.75 0.5; 0.25 0.5; 0.5 0.75; 0.875 0.375],
    )
    A1, A2, A12 =
        partial_nrm(ped, F[:, 1]), partial_nrm(ped, F[:, 2]), partial_nrm(ped, h12)
    check(
        "A₁, last row",
        A1[11, :],
        [0.375, 0.5, 0, 0, 0.812, 0.312, 0, 0.75, 0.156, 0.453, 1.188];
        atol = 1e-3,
    )
    check("A₁₂, last row", A12[11, :], [0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0.125, 0.375])

    # 3. Model 1: Breed-specific partial relationship model (García-Cortés & Toro, 2006)
    X = [first(incidence(herd)) F]
    ms = [nzinv(A) for A in (A1, A2, A12)]
    Zs = [Diagonal(Float64.(c .> 0)) for c in (F[:, 1], F[:, 2], h12)]
    m = mme(y, X, [RandomEffect(Zs[k], ms[k][1], σ²[k]) for k = 1:3]; σ²e)
    #    written out as T'T + Σ: each animal's record is linked to the breed
    #    components it carries (Zₖ selects animals with fₖ > 0), and Σ holds
    #    σ²e/σ²ₖ times the generalized inverses of A₁, A₂, A₁₂
    T = [X reduce(hcat, Zs)]
    Σ = blockdiag(
        spzeros(size(X, 2), size(X, 2)),
        (sparse(σ²e / σ²[k] * ms[k][1]) for k = 1:3)...,
    )
    check("Model 1: mme() LHS = T'T + Σ", m.lhs, T'T + Σ; atol = 1e-10)
    zero = [1; [m.blocks[k+1][ms[k][2]] for k = 1:3]...]
    b, u = solutions(m, solve_mme(T'T + Σ, T'y; zero))
    u = reduce(hcat, u)

    # 4. Model 2: Equivalent combined model with G = A₁σ²₁ + A₂σ²₂ + A₁₂σ²₁₂
    G = σ²[1] * A1 + σ²[2] * A2 + σ²[3] * A12
    #    one record per animal, so Z = I: [X'X X'; X I + σ²e G⁻¹]
    mc = mme(y, X, [RandomEffect(I(n), inv(G), 1.0)]; σ²e)
    lhs_c = [X'X X'; X I + σ²e * inv(G)]
    check("Model 2: mme() LHS = [X'X X'; X I + σ²e G⁻¹]", mc.lhs, lhs_c; atol = 1e-10)
    bc, (a,) = solutions(mc, solve_mme(lhs_c, [X'y; y]; zero = [1]))
    animals.u1, animals.u2, animals.u12 = eachcol(u)
    animals.a = vec(a)
    check("fixed effects (herd, breed)", [b bc], repeat([0, -3.198, 16.613, 17.408], 1, 2))
    check(
        "breed 1, breed 2, segregation",
        u,
        [
            -0.293 0 0;
            0.293 0 0;
            0 0.689 0;
            0 -0.689 0;
            0.237 0 0;
            0.388 0.828 0;
            0 0.671 0;
            0.750 0.705 0.291;
            0.230 0.962 0.071;
            0.556 0.966 0.257;
            0.781 0.441 0.234
        ];
        atol = 1.5e-3,
    )
    check("combined = sum of breed-specific", vec(a), vec(sum(u; dims = 2)); atol = 1e-8)

    # 5. Example 14.2: Random-regression approximation (Strandén & Mäntysaari, 2013)
    println("Example 14.2: random-regression approximation")
    A = nrm(ped)
    cs = (F[:, 1], F[:, 2], h12)
    ks = [nzinv(Diagonal(c .> 0) * A * Diagonal(c .> 0)) for c in cs]
    mr = mme(y, X, [RandomEffect(Diagonal(sqrt.(cs[k])), ks[k][1], σ²[k]) for k = 1:3]; σ²e)
    #    written out: covariables √f₁, √f₂, √h₁₂ in T, the same pedigree A restricted
    #    to the animals carrying each component in Σ
    Tr = [X reduce(hcat, [Diagonal(sqrt.(c)) for c in cs])]
    Σr = blockdiag(
        spzeros(size(X, 2), size(X, 2)),
        (sparse(σ²e / σ²[k] * ks[k][1]) for k = 1:3)...,
    )
    check("Example 14.2: mme() LHS = T'T + Σ", mr.lhs, Tr'Tr + Σr; atol = 1e-10)
    zr = [1; [mr.blocks[k+1][ks[k][2]] for k = 1:3]...]
    br, ur = solutions(mr, solve_mme(Tr'Tr + Σr, Tr'y; zero = zr))
    ur = reduce(hcat, ur)
    bv = [sum(sqrt(cs[k][i]) * ur[i, k] for k = 1:3) for i = 1:11]

    animals.a_rr = bv
    check("fixed effects", br, [0, -3.051, 16.504, 17.462])
    # Note: Table 14.3 in the book lists the contributions √fₚ uₚ that sum to a*
    check(
        "contributions √f₁u₁, √f₂u₂, √h₁₂u₁₂",
        reduce(hcat, [sqrt.(cs[k]) .* ur[:, k] for k = 1:3]),
        [
            -0.333 0 0;
            0.147 0 0;
            0 0.298 0;
            0 -0.726 0;
            0.104 0 0;
            0.437 0.782 0;
            0 0.427 0;
            0.721 0.792 0.375;
            0.257 0.838 0.190;
            0.535 1.019 0.390;
            0.723 0.542 0.332
        ],
    )
    check(
        "total breeding values a* = ΣFₚuₚ + Hu₁₂",
        bv,
        [-0.333, 0.147, 0.298, -0.726, 0.104, 1.219, 0.427, 1.888, 1.286, 1.944, 1.597],
    )

    show_table(animals)

    (; b, u, a = vec(a), br, ur, bv, animals)
end
