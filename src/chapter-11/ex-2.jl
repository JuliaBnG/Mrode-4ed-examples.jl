# Example 11.2 – SNP-BLUP (random SNP effects), unweighted and with EDC
# weights.

"""SNP-BLUP solutions (mean, SNP effects) for the reference bulls."""
function snpblup(Z, y, σ²a, σ²e, k; w = nothing)
    m = w === nothing ?
        mme(y, ones(length(y), 1), [RandomEffect(Z, I, σ²a / k)]; σ²e) :
        mme(y, ones(length(y), 1), [RandomEffect(Z, I, σ²a / k)]; Rinv = w ./ σ²e)
    solve_mme(m)
end

function ex_11_2()
    println("Example 11.2: SNP-BLUP")
    (; M, dyd, edc, ref, cand, σ²a, σ²e) = data_11()
    Z, p, k = center_genotypes(M)
    check("2Σpq and α", [k, k * σ²e / σ²a], [3.5383, 24.598]; atol = 2e-3)
    x = snpblup(Z[ref, :], dyd[ref], σ²a, σ²e, k)
    xw = snpblup(Z[ref, :], dyd[ref], σ²a, σ²e, k; w = edc[ref])
    show_table(effect = ["mean"; "SNP " .* string.(1:10)], unweighted = x, weighted = xw)
    check("unweighted: mean, SNP effects", x,
          [9.944, 0.087, -0.311, 0.262, -0.080, 0.110, 0.139, 0, 0, -0.061, -0.016])
    check("weighted: mean, SNP effects", xw,
          [11.876, -0.633, -3.041, 3.069, -1.267, 2.600, 4.447, 0, 0, -3.240, 1.883])
    check("unweighted DGV (reference, candidates)", Z * x[2:end],
          [0.070, 0.111, 0.045, 0.253, -0.495, -0.357, 0.145, -0.224, 0.027, 0.114, -0.240,
           0.143, 0.054, 0.354])
    check("weighted DGV (reference, candidates)", Z * xw[2:end],
          [-2.651, 1.307, 0.611, 1.007, -5.693, -4.358, 0.502, -5.718, -0.006, 6.513,
           -3.835, 2.701, 3.273, 6.350])
    (; x, xw)
end
