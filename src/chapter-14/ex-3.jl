# Examples 14.3 and 14.4 – purebred and crossbred performance with breed
# of origin of alleles (Christensen et al., 2014): pedigree-based
# breed-specific partial relationship matrices, then their genomic
# counterparts from phased crossbred genotypes.

"""
Breed-of-origin MME (Eqn 14.24): per breed `p`, a two-component random
effect (purebred performance, crossbred performance) with K⁻¹ = A⁽ᵖ⁾⁻¹
(or G⁽ᵖ⁾⁻¹): purebred-p records map to the animal, crossbred records to
½ × their breed-p parent.
"""
function boa(Ks, nulls)
    (; ped, group, herd, y, S1, S2, σ²e, σ²x) = data_14_3()
    n = length(y)
    X = reduce(hcat, [first(incidence(herd)) .* (group .== g) for g in 1:3])
    Zp = [incidence([group[i] == p ? i : 0 for i in 1:n], n) for p in 1:2]
    parent(i, p) = (s = ped.sire[i]; group[s] == p ? s : ped.dam[i])
    Zx = [incidence([group[i] == 3 ? parent(i, p) : 0 for i in 1:n], n; w = 0.5)
          for p in 1:2]
    rinv = [group[i] == 3 ? 1 / σ²x : 1 / σ²e for i in 1:n]
    m = mme(y, X, [RandomEffect([Zp[1], Zx[1]], Ks[1], S1),
                   RandomEffect([Zp[2], Zx[2]], Ks[2], S2)]; Rinv = rinv)
    zero = reduce(vcat, [m.blocks[k+1][[nulls[k]; nulls[k] .+ n]] for k in 1:2])
    b, (g1, g2) = solutions(m, solve_mme(m; zero))
    (; b, g1, g2)
end

function ex_14_3()
    println("Example 14.3: breed of origin of alleles, pedigree")
    (; ped, group) = data_14_3()
    founder = Float64[group .== 1 group .== 2] .* (ped.sire .== 0)
    F = breed_composition(ped, founder)
    A = [partial_nrm(ped, F[:, p]) for p in 1:2]
    check("A⁽¹⁾, last row", A[1][12, :], [0.25, 0.25, 0.5, 0.375, 0, 0, 0, 0, 0.125, 0.125,
                                          0.125, 0.5])
    check("A⁽²⁾, last row", A[2][12, :], [0, 0, 0, 0, 0.5, 0, 0.25, 0.375, 0, 0.125, 0.25,
                                          0.5])
    Ks = nzinv.(A)
    r = boa(first.(Ks), last.(Ks))
    show_table(animal = 1:12, breed1 = r.g1[:, 1], cross1 = r.g1[:, 2], breed2 = r.g2[:, 1],
               cross2 = r.g2[:, 2])
    # Solutions of Eqn 14.24 as stated agree with Table 14.5 to within 0.005;
    # they are exactly symmetric for the exchangeable founders (e.g. ±0.057
    # for animals 1 and 2) where the book's are not (−0.060, 0.055). The
    # same code reproduces the genomic Example 14.4 exactly.
    check("herd effects (breed 1, breed 2, crossbred)", r.b,
          [12.875, 11.924, 15.767, 16.899, 20.002, 20.887]; atol = 5e-3)
    check("breed 1 and crossbred 1",
          r.g1, [-0.060 -0.144; 0.055 0.138; 0.212 0.227; 0.195 0.151; 0 0; 0 0; 0 0; 0 0;
                 -0.028 -0.070; -0.028 -0.070; 0.028 0.070; 0.105 0.112]; atol = 5e-3)
    check("breed 2 and crossbred 2",
          r.g2, [0 0; 0 0; 0 0; 0 0; 0.200 0.212; -0.207 -0.216; 0.267 0.159; 0.409 0.307;
                 -0.101 -0.106; 0.132 0.078; 0.101 0.106; 0.101 0.106]; atol = 5e-3)
    check("breeding value of crossbred 12", r.g1[12, 2] + r.g2[12, 2], 0.21857; atol = 1e-4)

    println("Example 14.4: breed of origin of alleles, genomic")
    M = [2 0 1 1 0 0 2 2 1 2; 1 1 0 0 1 2 0 1 1 0; 2 1 1 0 0 1 1 1 1 1; 2 0 2 1 0 0 2 1 2 1;
         1 1 2 1 1 0 2 2 1 2; 0 0 1 1 0 1 0 1 2 1; 1 1 2 0 1 0 1 1 2 2; 2 0 2 1 0 0 2 1 1 2]
    Q = ([1 0 1 0 0 0 1 1 1 1; 1 0 1 1 0 0 1 1 1 1; 1 1 1 0 1 0 1 1 1 1; 1 1 1 1 0 0 1 1 1 1],
         [0 0 0 1 0 0 0 1 1 0; 1 1 1 0 0 0 1 1 1 1; 1 1 0 0 0 1 0 0 1 0; 1 0 1 0 0 1 0 1 0 1])
    pure = (1:4, 5:8)
    Ks, nulls = Matrix{Float64}[], Vector{Int}[]
    for p in 1:2
        Mp = M[pure[p], :]
        f = vec(sum([Mp; Q[p]]; dims = 1)) ./ (2 * 4 + 4)
        Zp = Mp .- 2f'
        Wp = Q[p] .- f'
        Gp = [Zp; Wp] * [Zp; Wp]' / (2sum(f .* (1 .- f)))
        # The G⁽¹⁾ printed in the book (e.g. g₁₁ = 1.815) does not follow from
        # Eqn 14.25 (g₁₁ = 0.607); Table 14.9 does, and is reproduced below.
        p == 1 && check("breed 1 allele frequencies", f, [0.917, 0.333, 0.667, 0.333, 0.167,
                                                         0.250, 0.750, 0.750, 0.750, 0.667])
        idx = [collect(pure[p]); 9:12]
        K = zeros(12, 12)
        K[idx, idx] = inv(Gp + 0.01I)
        push!(Ks, K)
        push!(nulls, collect(pure[3-p]))
    end
    s = boa(Ks, nulls)
    show_table(animal = 1:12, breed1 = s.g1[:, 1], cross1 = s.g1[:, 2], breed2 = s.g2[:, 1],
               cross2 = s.g2[:, 2])
    check("herd effects", s.b, [12.955, 12.054, 15.801, 17.062, 20.070, 20.940])
    check("breed 1 and crossbred 1",
          s.g1, [-0.177 -0.282; -0.062 0.212; 0.070 0.164; 0.151 0.080; 0 0; 0 0; 0 0; 0 0;
                 0.015 -0.037; -0.026 -0.104; 0.014 0.005; 0.015 -0.037])
    check("breed 2 and crossbred 2",
          s.g2, [0 0; 0 0; 0 0; 0 0; 0.176 0.208; -0.485 -0.419; 0.221 0.150; 0.360 0.302;
                 -0.364 -0.281; 0.213 0.178; -0.117 -0.135; -0.005 -0.003])
    (; r, s)
end
