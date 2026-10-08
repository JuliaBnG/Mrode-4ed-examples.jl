# Example 11.6 – base-population allele frequencies and gene content of
# ungenotyped ancestors by the linear method (Gengler et al., 2007).

function ex_11_6()
    println("Example 11.6: base allele frequencies and imputed gene content")
    (; ped, M, id) = data_11()
    Ai = ainv(ped)
    p, μ, d = base_allele_frequencies(M, Ai, id; λ = 0.01)
    q = μ[1] .+ d[:, 1]                           # SNP 1
    show_table(animal = 1:26, BLUP = d[:, 1], predicted = q, rounded = round.(Int, q))
    check("mean of SNP 1 (= 2p̂)", μ[1], 0.597)
    check("BLUPs of SNP 1", d[:, 1],
          [0.695, -0.523, 0.695, -0.198, -0.523, 0.140, -0.518, -0.518, -0.779, 0.140,
           -0.518, 0.140, 1.387, 0.380, 0.397, -0.586, -0.586, 0.400, -0.589, -0.589, 1.389,
           -0.587, -0.587, 0.400, -0.587, 0.400]; atol = 1.5e-3)
    check("rounded genotypes of genotyped bulls = true", round.(Int, q[id]), M[:, 1];
          atol = 0)
    # GLS estimate (Garcia-Baccino et al., 2017). The book reports 0.575,
    # which neither the genotyped block of A⁻¹ nor the inverse of A₂₂
    # reproduces; the latter is the GLS estimator and is close to μ̂ = 0.597.
    o = ones(14)
    gls(K) = (o' * K * M[:, 1]) / (o' * K * o)
    @printf("  GLS mean: with (A⁻¹)₂₂ %.3f, with (A₂₂)⁻¹ %.3f (book 0.575)\n",
            gls(Matrix(Ai[id, id])), gls(inv(nrm(ped, id))))
    println("  base allele frequencies of all SNPs: ", round.(p; digits = 3))
    (; p, μ, d)
end
