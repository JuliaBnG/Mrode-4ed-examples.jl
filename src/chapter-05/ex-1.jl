# Example 5.1 – repeatability model, and daughter yield deviation (5.2.3).

function ex_05_1()
    println("Example 5.1: repeatability model, fat yield")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 3, 1], dam = [0, 0, 0, 2, 2, 5, 4, 7])
    cow = repeat(4:8; inner = 2)
    parity = repeat([1, 2], 5)
    hys = [1, 3, 1, 4, 2, 3, 1, 3, 2, 4]
    fy = [201, 280, 150, 200, 160, 190, 180, 250, 285, 300.0]
    σ²a, σ²pe, σ²e = 20.0, 12.0, 28.0

    X = [first(incidence(hys)) first(incidence(parity))]
    Z = incidence(cow, nrow(ped))
    W, _ = incidence(cow)                     # pe for cows 4–8
    m = mme(fy, X, [RandomEffect(Z, ainv(ped), σ²a; name = "animal"),
                    RandomEffect(W, I, σ²pe; name = "pe")]; σ²e)
    x = solve_mme(m; zero = [1, 3])           # HYS 1 and 3 set to zero
    b, (a, pe) = solutions(m, x)
    a, pe = vec(a), vec(pe)
    show_table(effect = ["HYS " .* string.(1:4); "parity " .* string.(1:2);
                         "animal " .* string.(1:8); "pe " .* string.(4:8)], solution = x)
    check("HYS and parity", b, [0, 44.065, 0, 0.013, 175.472, 241.893])
    check("animals", a, [10.148, -3.084, -7.063, 13.581, -18.207, -18.387, 9.328, 24.194])
    check("permanent environment", pe, [8.417, -7.146, -17.229, -1.390, 17.347])

    # 5.2.3 DYD of sire 1: YD adjusted for fixed and pe effects
    yadj = fy - X * b - W * pe
    nrec = zeros(Int, 8)
    yd = zeros(8)
    for (k, c) in enumerate(cow)
        nrec[c] += 1
        yd[c] += yadj[k]
    end
    yd[4:8] ./= nrec[4:8]
    p = ebv_partition(ped.sire, ped.dam, σ²e / σ²a, nrec, yd, a)
    check("YD of cows 4, 6 and 8", yd[[4, 6, 8]], [23.4005, -38.486, 44.432]; atol = 2e-3)
    check("DYD of sire 1", p.DYD[1], 23.552; atol = 2e-3)
    (; b, a, pe, partition = p)
end
