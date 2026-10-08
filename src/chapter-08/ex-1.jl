# Example 8.1 – animal model for a maternal trait: direct and maternal
# genetic effects (correlated) and maternal permanent environment.

"""Data of Table 8.1 (birth weight of beef calves)."""
function data_08()
    ped = DataFrame(sire = [0, 0, 0, 0, 1, 3, 4, 3, 1, 3, 3, 8, 9, 3],
                    dam = [0, 0, 0, 0, 2, 2, 6, 5, 6, 2, 7, 7, 2, 6])
    calf = 5:14
    herd = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    pen = [1, 2, 2, 1, 1, 2, 2, 2, 1, 2]
    bw = [35.0, 20, 25, 40, 42, 22, 35, 34, 20, 40]
    G0 = [150.0 -40; -40 90]
    (; ped, calf, herd, pen, bw, G0, σ²pe = 40.0, σ²e = 350.0)
end

function ex_08_1()
    println("Example 8.1: maternal animal model, birth weight")
    (; ped, calf, herd, pen, bw, G0, σ²pe, σ²e) = data_08()
    dam = ped.dam[calf]
    X = [first(incidence(herd)) first(incidence(pen))]
    Z = incidence(calf, nrow(ped))          # direct
    W = incidence(dam, nrow(ped))           # maternal genetic, through the dam
    S, dams = incidence(dam)                # maternal pe of dams 2, 5, 6, 7
    m = mme(bw, X, [RandomEffect([Z, W], ainv(ped), G0; name = "direct+maternal"),
                    RandomEffect(S, I, σ²pe; name = "pe")]; σ²e)
    check("variance ratios α", σ²e * inv(G0), [2.647 1.176; 1.176 4.412])
    x = solve_mme(m; zero = [1])            # herd 1 set to zero
    b, (um, pe) = solutions(m, x)
    show_table(animal = 1:14, direct = um[:, 1], maternal = um[:, 2])
    check("herd and pen", b, [0, 3.386, 1.434, 34.540, 27.691])
    check("direct", um[:, 1], [0.564, -1.244, 1.165, -0.484, 0.630, -0.859, -1.156, 1.917,
                               -0.553, -1.055, 0.385, 0.863, -2.980, 1.751])
    check("maternal", um[:, 2], [0.262, -1.583, 0.736, 0.586, -0.507, 0.841, 1.299, -0.158,
                                 0.660, -0.153, 0.916, 0.442, 0.093, 0.362])
    check("permanent environment (dams $(dams))", pe, [-1.701, 0.415, 0.825, 0.461])
    check("non-zeros in the coefficient matrix", nnz(m.lhs), 429)
    (; b, um, pe)
end
