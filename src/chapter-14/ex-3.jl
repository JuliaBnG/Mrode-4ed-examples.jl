"""
    boa(Ks, nulls) -> NamedTuple

Fit the Breed-of-Origin of Alleles (BOA) MME (Eqn 14.24; Christensen et al., 2014): For each
breed ``p \\in \\{1, 2\\}``, fits a two-trait random effect (purebred performance
``u^{(p)}``, crossbred performance ``u_c^{(p)}``) with relationship inverse
``K^{-1} = (A^{(p)})^{-1}`` (pedigree) or ``(G^{(p)})^{-1}`` (genomic). Purebred-``p``
records map directly to the animal (``Z_p``), while crossbred records map to
``\\frac{1}{2}`` times their breed-``p`` parent (``Z_x``).
"""
function boa(Ks, nulls)
    (; ped, group, herd, y, S1, S2, σ²e, σ²x) = data_14_3()
    n = length(y)
    X = reduce(hcat, [first(incidence(herd)) .* (group .== g) for g = 1:3])
    Zp = [incidence([group[i] == p ? i : 0 for i = 1:n], n) for p = 1:2]
    parent(i, p) = (s = ped.sire[i]; group[s] == p ? s : ped.dam[i])
    Zx = [incidence([group[i] == 3 ? parent(i, p) : 0 for i = 1:n], n; w = 0.5) for p = 1:2]
    rinv = [group[i] == 3 ? 1 / σ²x : 1 / σ²e for i = 1:n]
    m = mme(
        y,
        X,
        [RandomEffect([Zp[1], Zx[1]], Ks[1], S1), RandomEffect([Zp[2], Zx[2]], Ks[2], S2)];
        Rinv = rinv,
    )
    #    written out: crossbred records have their own residual variance, so the
    #    MME are in the general R⁻¹ form with W = [Z₁ Zₓ₁ Z₂ Zₓ₂] and, per breed,
    #    the two-trait block S⁻¹ ⊗ K⁻¹ (purebred and crossbred performance)
    Ri = Diagonal(rinv)
    W = [Zp[1] Zx[1] Zp[2] Zx[2]]
    Σ = blockdiag(sparse(kron(inv(S1), Ks[1])), sparse(kron(inv(S2), Ks[2])))
    #! format: off
    lhs = [
        X'Ri*X  X'Ri*W
        W'Ri*X  W'Ri*W + Σ
    ]
    #! format: on
    check("BOA: mme(; Rinv) LHS = general form", m.lhs, lhs; atol = 1e-10)
    zero = reduce(vcat, [m.blocks[k+1][[nulls[k]; nulls[k] .+ n]] for k = 1:2])
    b, (g1, g2) = solutions(m, solve_mme(lhs, [X'Ri*y; W'Ri*y]; zero))
    (; b, g1, g2)
end

"""
    ex_14_3() -> NamedTuple

Examples 14.3 & 14.4: Breed-of-Origin of Alleles (BOA) in crossbreeding (pedigree and
genomic) (Mrode & Pocrnic, 2023, Section 14.7; Christensen et al., 2014).

# Model Formulation
In terminal two-breed crossbreeding, alleles in crossbred progeny originate from distinct
purebred lines.
1.  **Pedigree-based BOA** (Example 14.3): Breed-specific partial relationship matrices
   ``A^{(1)}`` and ``A^{(2)}`` model the purebred and crossbred additive genetic merit
   transmitted from each parental breed.
2.  **Genomic BOA** (Example 14.4): Crossbred genotypes are phased and alleles assigned to
   breed of origin. Genomic relationship matrices ``G^{(1)}`` and ``G^{(2)}`` are
   constructed using breed-specific allele frequencies ``f^{(1)}`` and ``f^{(2)}`` (Eqn
   14.25).

# Key Concepts Illustrated
1. **One table for purebred and crossbred animals**: All 12 animals (4 purebred breed 1,
   4 purebred breed 2, and 4 F1 crossbreds) are kept in `animals` with their breed group,
   herd, and phenotype; the purebred (`u1`, `u2`) and crossbred (`uc1`, `uc2`) breeding
   values from both the pedigree and genomic BOA models are added as columns.
2. **Heterogeneous residuals and the general ``R^{-1}`` MME**: In a reduced model where
   crossbred Mendelian sampling is absorbed into the residual, crossbred records have a
   larger residual variance (``\\sigma_x^2 = 4.75``) than purebred records
   (``\\sigma_e^2 = 4.0``). We therefore write the MME in the general form
   ``[X'R^{-1}X\\ X'R^{-1}W;\\ W'R^{-1}X\\ W'R^{-1}W + \\Sigma]``, with
   ``W = [Z_1\\ Z_{x1}\\ Z_2\\ Z_{x2}]`` (where ``Z_{xp}`` has ``1/2`` in the column of the
   crossbred's breed-``p`` parent) and ``\\Sigma = \\mathrm{blockdiag}(S_1^{-1} \\otimes
   K_1^{-1},\\, S_2^{-1} \\otimes K_2^{-1})``, and verify it against `mme`.
3. **Genomic breed-of-origin matrix ``G^{(p)}``** (Eqn 14.25): Built from centred purebred
   genotypes ``Z_p = M_p - 2f_p'`` and phased crossbred breed-``p`` alleles
   ``W_p = Q_p - f_p'`` as ``G^{(p)} = [Z_p;\\ W_p][Z_p;\\ W_p]' / 2\\sum f_j(1 - f_j)``.

# Notes on Text Discrepancies
1. Example 14.3: Solutions agree with Table 14.5 within 0.005. The code here yields exact
   symmetry for exchangeable founders (e.g. ``\\pm 0.057`` for animals 1 and 2), whereas the
   book has a slight rounding asymmetry (``-0.060``, ``0.055``).
2. Example 14.4: The ``G^{(1)}`` matrix printed in the book does not match Eqn 14.25 (prints
   ``g_{11} = 1.815`` instead of ``0.607``), but Table 14.9 follows Eqn 14.25 exactly and is
   reproduced here.

# Returns
A `NamedTuple` with fields:
- `animals`: The animal table with pedigree (`u1`, `uc1`, `u2`, `uc2`) and genomic (`gu1`,
  `guc1`, `gu2`, `guc2`) BOA solutions added.
- `r`: Results from the pedigree BOA model (Example 14.3).
- `s`: Results from the genomic BOA model (Example 14.4).
"""
function ex_14_3()
    println("Example 14.3: breed of origin of alleles, pedigree")

    # 1. Pedigree BOA (Example 14.3)
    (; ped, animals, group) = data_14_3()
    founder = Float64[group .== 1 group .== 2] .* (ped.sire .== 0)
    F = breed_composition(ped, founder)
    A = [partial_nrm(ped, F[:, p]) for p = 1:2]
    check(
        "A⁽¹⁾, last row",
        A[1][12, :],
        [0.25, 0.25, 0.5, 0.375, 0, 0, 0, 0, 0.125, 0.125, 0.125, 0.5],
    )
    check(
        "A⁽²⁾, last row",
        A[2][12, :],
        [0, 0, 0, 0, 0.5, 0, 0.25, 0.375, 0, 0.125, 0.25, 0.5],
    )
    Ks = nzinv.(A)
    r = boa(first.(Ks), last.(Ks))

    animals.u1, animals.uc1, animals.u2, animals.uc2 = eachcol([r.g1 r.g2])
    check(
        "herd effects (breed 1, breed 2, crossbred)",
        r.b,
        [12.875, 11.924, 15.767, 16.899, 20.002, 20.887];
        atol = 5e-3,
    )
    check(
        "breed 1 and crossbred 1",
        r.g1,
        [
            -0.060 -0.144;
            0.055 0.138;
            0.212 0.227;
            0.195 0.151;
            0 0;
            0 0;
            0 0;
            0 0;
            -0.028 -0.070;
            -0.028 -0.070;
            0.028 0.070;
            0.105 0.112
        ];
        atol = 5e-3,
    )
    check(
        "breed 2 and crossbred 2",
        r.g2,
        [
            0 0;
            0 0;
            0 0;
            0 0;
            0.200 0.212;
            -0.207 -0.216;
            0.267 0.159;
            0.409 0.307;
            -0.101 -0.106;
            0.132 0.078;
            0.101 0.106;
            0.101 0.106
        ];
        atol = 5e-3,
    )
    check("breeding value of crossbred 12", r.g1[12, 2] + r.g2[12, 2], 0.21857; atol = 1e-4)

    # 2. Genomic BOA (Example 14.4)
    println("Example 14.4: breed of origin of alleles, genomic")
    M = [
        2 0 1 1 0 0 2 2 1 2;
        1 1 0 0 1 2 0 1 1 0;
        2 1 1 0 0 1 1 1 1 1;
        2 0 2 1 0 0 2 1 2 1;
        1 1 2 1 1 0 2 2 1 2;
        0 0 1 1 0 1 0 1 2 1;
        1 1 2 0 1 0 1 1 2 2;
        2 0 2 1 0 0 2 1 1 2
    ]
    Q = (
        [
            1 0 1 0 0 0 1 1 1 1;
            1 0 1 1 0 0 1 1 1 1;
            1 1 1 0 1 0 1 1 1 1;
            1 1 1 1 0 0 1 1 1 1
        ],
        [
            0 0 0 1 0 0 0 1 1 0;
            1 1 1 0 0 0 1 1 1 1;
            1 1 0 0 0 1 0 0 1 0;
            1 0 1 0 0 1 0 1 0 1
        ],
    )
    pure = (1:4, 5:8)
    Ks_g, nulls_g = Matrix{Float64}[], Vector{Int}[]
    for p = 1:2
        Mp = M[pure[p], :]
        f = vec(sum([Mp; Q[p]]; dims = 1)) ./ (2 * 4 + 4)
        Zp = Mp .- 2f'
        Wp = Q[p] .- f'
        Gp = [Zp; Wp] * [Zp; Wp]' / (2sum(f .* (1 .- f)))
        p == 1 && check(
            "breed 1 allele frequencies",
            f,
            [0.917, 0.333, 0.667, 0.333, 0.167, 0.250, 0.750, 0.750, 0.750, 0.667],
        )
        idx = [collect(pure[p]); 9:12]
        K = zeros(12, 12)
        K[idx, idx] = inv(Gp + 0.01I)
        push!(Ks_g, K)
        push!(nulls_g, collect(pure[3-p]))
    end

    s = boa(Ks_g, nulls_g)
    animals.gu1, animals.guc1, animals.gu2, animals.guc2 = eachcol([s.g1 s.g2])
    println("\nPurebred (u) and crossbred (uc) performance of breed-1 and breed-2 alleles,")
    println("pedigree (Example 14.3) and genomic (g…, Example 14.4):")
    show_table(animals)
    check("herd effects", s.b, [12.955, 12.054, 15.801, 17.062, 20.070, 20.940])
    check(
        "breed 1 and crossbred 1",
        s.g1,
        [
            -0.177 -0.282;
            -0.062 0.212;
            0.070 0.164;
            0.151 0.080;
            0 0;
            0 0;
            0 0;
            0 0;
            0.015 -0.037;
            -0.026 -0.104;
            0.014 0.005;
            0.015 -0.037
        ],
    )
    check(
        "breed 2 and crossbred 2",
        s.g2,
        [
            0 0;
            0 0;
            0 0;
            0 0;
            0.176 0.208;
            -0.485 -0.419;
            0.221 0.150;
            0.360 0.302;
            -0.364 -0.281;
            0.213 0.178;
            -0.117 -0.135;
            -0.005 -0.003
        ],
    )

    (; r, s, animals)
end
