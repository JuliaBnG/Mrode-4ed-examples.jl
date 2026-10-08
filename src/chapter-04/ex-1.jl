# Example 4.1 – univariate animal model, accuracy of evaluations (4.3.3)
# and partition of breeding values into PA, YD, PC and DYD (4.3.1–4.3.2).

function ex_04_1()
    println("Example 4.1: animal model, pre-weaning gain of beef calves")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    wwg = [4.5, 2.9, 3.9, 3.5, 5.0]
    σ²a, σ²e = 20.0, 40.0
    α = σ²e / σ²a

    X, _ = incidence(sex; levels = ["M", "F"])
    Z = incidence(calf, nrow(ped))
    m = mme(wwg, X, [RandomEffect(Z, ainv(ped), σ²a; name = "animal")]; σ²e)
    x = solve_mme(m)
    b, (a,) = solutions(m, x)
    a = vec(a)
    show_table(effect = ["male", "female", string.(1:8)...], solution = x)
    check("sex effects", b, [4.358, 3.404])
    check("breeding values", a,
          [0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.249, 0.183])

    # 4.3.3 accuracy: PEV = C²² σ²e, r² = 1 − dᵢα
    C = lhs_inverse(m)
    d = diag(C)[m.blocks[2]]
    r² = 1 .- d .* α
    sep = sqrt.(d .* σ²e)
    show_table(animal = 1:8, d = d, r2 = r², r = sqrt.(r²), SEP = sep)
    check("diagonals of C⁻¹", d, [0.471, 0.492, 0.456, 0.428, 0.428, 0.442, 0.442, 0.422])
    check("reliabilities", r², [0.058, 0.016, 0.088, 0.144, 0.144, 0.116, 0.116, 0.156])
    check("SEP", sep, [4.341, 4.436, 4.271, 4.138, 4.138, 4.205, 4.205, 4.109]; atol = 3e-3)
    check("reliability() without inverting C", vec(reliability(m, 1)), r²; atol = 1e-10)

    # 4.3.1–4.3.2 EBV = n₁PA + n₂YD + n₃PC, and progeny yield deviation
    nrec = zeros(Int, 8)
    nrec[calf] .= 1
    yd = zeros(8)
    yd[calf] = wwg - X * b
    p = ebv_partition(ped.sire, ped.dam, α, nrec, yd, a)
    show_table(animal = 1:8, PA = p.PA, YD = p.YD, PC = p.PC, n1 = p.n1, n2 = p.n2,
               n3 = p.n3, EBV = p.n1 .* p.PA .+ p.n2 .* p.YD .+
                              p.n3 .* replace(p.PC, NaN => 0.0))
    check("progeny yield deviation of sire 3", p.DYD[3], 0.059)
    (; b, a, C, partition = p)
end
