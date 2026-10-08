# Example 8.2 – reduced animal model with maternal effects, and
# back-solving direct and maternal effects of non-parents (Eqns 8.8, 8.10).

function ex_08_2()
    println("Example 8.2: reduced animal model with maternal effects")
    (; ped, calf, herd, pen, bw, G0, σ²pe, σ²e) = data_08()
    dam = ped.dam[calf]
    X = [first(incidence(herd)) first(incidence(pen))]
    Z1, rinv, parents, Api = ram_design(ped, calf, G0[1, 1], σ²e)
    col = zeros(Int, nrow(ped))
    col[parents] = eachindex(parents)
    Z2 = incidence(col[dam], length(parents))   # dams are parents
    S, _ = incidence(dam)
    m = mme(bw, X, [RandomEffect([Z1, Z2], Api, G0),
                    RandomEffect(S, I, σ²pe)]; Rinv = rinv)
    x = solve_mme(m; zero = [1])
    b, (um, pe) = solutions(m, x)
    check("R⁻¹", rinv, [fill(0.00286, 5); fill(0.00235, 5)]; atol = 1e-5)
    check("non-zeros in the coefficient matrix", nnz(m.lhs), 329)

    # back-solving: û from PA and Mendelian sampling, m̂ by regression on it
    yadj = bw - X * b - Z2 * um[:, 2] - S * vec(pe)
    u = ram_backsolve(ped, calf, yadj, um[:, 1], parents, G0[1, 1], σ²e)
    mat = fill(NaN, nrow(ped))
    mat[parents] = um[:, 2]
    for i in 1:nrow(ped)
        isnan(mat[i]) || continue
        s, d = ped.sire[i], ped.dam[i]
        pa(v) = 0.5 * ((s > 0 ? v[s] : 0.0) + (d > 0 ? v[d] : 0.0))
        mat[i] = pa(mat) + G0[1, 2] / G0[1, 1] * (u[i] - pa(u))
    end
    show_table(animal = 1:14, direct = u, maternal = mat)
    ref = ex_08_1()
    check("RAM = animal model, fixed effects", b, ref.b; atol = 1e-8)
    check("RAM = animal model, direct (all animals)", u, ref.um[:, 1]; atol = 1e-8)
    check("RAM = animal model, maternal (all animals)", mat, ref.um[:, 2]; atol = 1e-8)
    (; b, u, mat, pe)
end
