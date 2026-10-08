# Examples 14.1 and 14.2 – multibreed animal model with breed-specific
# partial relationship matrices (García-Cortés and Toro, 2006), the
# equivalent combined model, and the random-regression approximation of
# Strandén and Mäntysaari (2013).

function ex_14_1()
    println("Example 14.1: breed-specific and combined multibreed models")
    (; ped, F, herd, y, σ², σ²e) = data_14_1()
    h12 = segregation_coefficients(ped, F, 1, 2)
    check("f₁ and h₁₂", [F[:, 1] h12],
          [1 0; 1 0; 0 0; 0 0; 1 0; 0.5 0; 0 0; 0.75 0.5; 0.25 0.5; 0.5 0.75; 0.875 0.375])
    A1, A2, A12 = partial_nrm(ped, F[:, 1]), partial_nrm(ped, F[:, 2]), partial_nrm(ped, h12)
    check("A₁, last row", A1[11, :], [0.375, 0.5, 0, 0, 0.812, 0.312, 0, 0.75, 0.156, 0.453,
                                      1.188]; atol = 1e-3)
    check("A₁₂, last row", A12[11, :], [0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0.125, 0.375])
    X = [first(incidence(herd)) F]
    sol = Vector{Float64}[]
    ms = [nzinv(A) for A in (A1, A2, A12)]
    Zs = [Diagonal(Float64.(c .> 0)) for c in (F[:, 1], F[:, 2], h12)]
    m = mme(y, X, [RandomEffect(Zs[k], ms[k][1], σ²[k]) for k in 1:3]; σ²e)
    zero = [1; [m.blocks[k+1][ms[k][2]] for k in 1:3]...]
    b, u = solutions(m, solve_mme(m; zero))
    u = reduce(hcat, u)
    # combined model with G = A₁σ²₁ + A₂σ²₂ + A₁₂σ²₁₂
    G = σ²[1] * A1 + σ²[2] * A2 + σ²[3] * A12
    mc = mme(y, X, [RandomEffect(I(11), inv(G), 1.0)]; σ²e)
    bc, (a,) = solutions(mc, solve_mme(mc; zero = [1]))
    show_table(animal = 1:11, combined = vec(a), breed1 = u[:, 1], breed2 = u[:, 2],
               segregation = u[:, 3])
    check("fixed effects (herd, breed)", [b bc], repeat([0, -3.198, 16.613, 17.408], 1, 2))
    check("breed 1, breed 2, segregation", u,
          [-0.293 0 0; 0.293 0 0; 0 0.689 0; 0 -0.689 0; 0.237 0 0; 0.388 0.828 0;
           0 0.671 0; 0.750 0.705 0.291; 0.230 0.962 0.071; 0.556 0.966 0.257;
           0.781 0.441 0.234]; atol = 1.5e-3)
    check("combined = sum of breed-specific", vec(a), vec(sum(u; dims = 2)); atol = 1e-8)

    println("Example 14.2: random-regression approximation")
    A = nrm(ped)
    cs = (F[:, 1], F[:, 2], h12)
    ks = [nzinv(Diagonal(c .> 0) * A * Diagonal(c .> 0)) for c in cs]
    mr = mme(y, X, [RandomEffect(Diagonal(sqrt.(cs[k])), ks[k][1], σ²[k]) for k in 1:3]; σ²e)
    zr = [1; [mr.blocks[k+1][ks[k][2]] for k in 1:3]...]
    br, ur = solutions(mr, solve_mme(mr; zero = zr))
    ur = reduce(hcat, ur)
    bv = [sum(sqrt(cs[k][i]) * ur[i, k] for k in 1:3) for i in 1:11]
    show_table(animal = 1:11, total = bv, breed1 = ur[:, 1], breed2 = ur[:, 2],
               segregation = ur[:, 3])
    check("fixed effects", br, [0, -3.051, 16.504, 17.462])
    # the book tabulates the breed contributions √fₚ uₚ (they add up to a*)
    check("contributions √f₁u₁, √f₂u₂, √h₁₂u₁₂", reduce(hcat, [sqrt.(cs[k]) .* ur[:, k] for k in 1:3]),
          [-0.333 0 0; 0.147 0 0; 0 0.298 0; 0 -0.726 0; 0.104 0 0; 0.437 0.782 0;
           0 0.427 0; 0.721 0.792 0.375; 0.257 0.838 0.190; 0.535 1.019 0.390;
           0.723 0.542 0.332])
    check("total breeding values a* = ΣFₚuₚ + Hu₁₂", bv,
          [-0.333, 0.147, 0.298, -0.726, 0.104, 1.219, 0.427, 1.888, 1.286, 1.944, 1.597])
    (; b, u, a, br, ur, bv)
end
