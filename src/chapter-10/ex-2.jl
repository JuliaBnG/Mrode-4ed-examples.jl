# Example 10.2 – random regression test-day model: Legendre polynomials
# of order 2 for animal and pe effects; 305-day values, partition of
# random regressions (10.3.2) and reliabilities (10.3.4).

function ex_10_2()
    println("Example 10.2: random regression model")
    (; ped, dims, cow, dim, htd, y, Φ) = data_10()
    G = [3.297 0.594 -1.381; 0.594 0.921 -0.289; -1.381 -0.289 1.005]
    P = [6.872 -0.254 -1.101; -0.254 3.171 0.167; -1.101 0.167 2.457]
    σ²e = 3.710
    N = nrow(ped)

    X1, _ = incidence(htd)
    X2 = Φ[dim, :]
    Z = incidence(cow, N)
    Q, cows = incidence(cow)
    ϕ = Φ[dim, 1:3]                       # random-regression covariables
    m = mme(y, [X1 X2], [RandomEffect([Diagonal(ϕ[:, c]) * Z for c in 1:3], ainv(ped), G),
                         RandomEffect([Diagonal(ϕ[:, c]) * Q for c in 1:3], I, P)]; σ²e)
    x = solve_mme(m; zero = [10])
    b, (u, pe) = solutions(m, x)
    t106, t140 = legendre([106, 140], 3; tmin = 4, tmax = 310) |> eachrow
    check("genetic variance at DIM 106, covariance 106–140",
          [t106' * G * t106, t106' * G * t140], [2.6433, 3.0219]; atol = 1e-3)
    check("HTD", b[1:10], [10.0862, 7.5908, 8.5601, 8.2430, 6.3161, 3.0101, 3.1085, 3.1718,
                           0.5044, 0]; atol = 2e-4)
    check("fixed regressions", b[11:15], [16.6384, -0.6253, -0.1346, 0.3479, -0.4218];
          atol = 1e-4)
    check("animal regressions", u,
          [-0.0583 0.0552 -0.0442; -0.0728 -0.0305 -0.0244; 0.1311 -0.0247 0.0686;
           0.3445 0.0063 -0.3164; -0.4537 -0.0520 0.2798; -0.5485 0.0730 0.1946;
           0.8518 -0.0095 -0.3131; 0.2209 0.0127 -0.0174]; atol = 1e-4)
    check("pe regressions", pe,
          [-0.6487 -0.3601 -1.4718; -0.7761 0.1370 0.9688; -1.9927 0.9851 -0.0693;
           3.5188 -1.0510 -0.4048; -0.1013 0.2889 0.9771]; atol = 1e-4)

    # 305-day values: t = Σ_{d=6}^{310} φ(d)
    t = vec(sum(legendre(6:310, 3; tmin = 4, tmax = 310); dims = 1))
    # the book sums Φ built from 4-decimal coefficients (e.g. 0.7906 for
    # √(5/2)/2) over 305 days, hence its third element −1.5561 vs −1.5451
    check("t (days 6–310)", t, [215.6655, 2.4414, -1.5561]; atol = 0.012)
    show_table(animal = 1:N, BV305 = u * t)
    check("305-day breeding values", u * t,
          [-12.3731, -15.7347, 28.1078, 74.8132, -98.4153, -118.4265, 184.1701, 47.6907];
          atol = 0.03)
    check("305-day pe", pe * t, [-138.4887, -168.5531, -427.2378, 756.9415, -22.6619];
          atol = 0.03)

    # 10.3.4 reliability of k'u with k = t: PEV = t'Cⁱⁱt
    C = lhs_inverse(m; zero = [10]) * m.scale
    r = m.blocks[2]
    g = t' * G * t
    rel = [1 - (idx = r[(0:2) .* N .+ i]; t' * C[idx, idx] * t) / g for i in 1:N]
    # k'Pk is printed as 323462.969, but the book's own t and P give 320126
    check("k'Gk", g, 154896.766; atol = 25)
    @printf("  k'Pk = %.1f\n", t' * P * t)
    check("reliabilities", rel, [0.09, 0.04, 0.07, 0.12, 0.15, 0.06, 0.10, 0.05];
          atol = 0.005)

    # 10.3.2 partition: YD = (Q'R⁻¹Q)⁻¹Q'R⁻¹(y − Xb − Zpe)
    yc = y - [X1 X2] * b - sum(Diagonal(ϕ[:, c]) * Q * pe[:, c] for c in 1:3)
    ZRZ = [zeros(3, 3) for _ in 1:N]
    yd = zeros(N, 3)
    for i in cows
        k = cow .== i
        ZRZ[i] = ϕ[k, :]' * ϕ[k, :] / σ²e
        yd[i, :] = (ϕ[k, :]' * ϕ[k, :]) \ (ϕ[k, :]' * yc[k])
    end
    p = ebv_partition_mt(ped.sire, ped.dam, G, ZRZ, yd, u)
    # Q'Q of cow 6 is ill-conditioned: the book's 4-decimal yc moves YD by
    # ~0.003, i.e. ~0.7 kg on the 305-day scale (−1086.645 in the book)
    check("YD of cow 6", yd[6, :], [-5.0004, -4.6419, -1.9931]; atol = 5e-3)
    check("YD of cow 6, 305-day value", t' * yd[6, :], -1086.6450; atol = 0.8)
    check("cow 6: W₁ (row 2)", p.W1[6][2, :], [0.0935, 0.8107, 0.2402]; atol = 1e-4)
    check("cow 6: W₂ (row 2)", p.W2[6][2, :], [-0.0935, 0.1893, -0.2402]; atol = 1e-4)
    PAs = sum(Diagonal(t) * p.W1[6] * p.PA[6])
    YDs = sum(Diagonal(t) * p.W2[6] * yd[6, :])
    check("cow 6: 305-day contributions of PA and YD", [PAs, YDs], [-26.4049, -91.9862];
          atol = 0.05)
    check("cow 4: W₁PA + W₂YD + W₃PC = û₄",
          p.W1[4] * p.PA[4] + p.W2[4] * yd[4, :] + p.W3[4] * p.PC[4], u[4, :]; atol = 1e-10)
    (; b, u, pe, rel)
end
