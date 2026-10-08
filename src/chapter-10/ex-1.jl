# Example 10.1 – fixed regression test-day model: one Legendre lactation
# curve of order 4, animal and permanent environment effects.

function ex_10_1()
    println("Example 10.1: fixed regression model for test-day fat yield")
    (; ped, dims, cow, dim, htd, y, Φ) = data_10()
    σ²a, σ²pe, σ²e = 5.521, 8.470, 3.710
    check("Φ (Appendix G), first and last rows", Φ[[1, 10], :],
          [0.7071 -1.2247 1.5811 -1.8704 2.1213; 0.7071 1.2247 1.5811 1.8704 2.1213];
          atol = 5e-4)  # the book rounds the Legendre coefficients to 4 decimals

    X1, _ = incidence(htd)
    X2 = Φ[dim, :]                                  # fixed lactation curve
    Z = incidence(cow, nrow(ped))
    Q, _ = incidence(cow)
    m = mme(y, [X1 X2], [RandomEffect(Z, ainv(ped), σ²a), RandomEffect(Q, I, σ²pe)]; σ²e)
    check("X₂'X₂", Matrix(m.lhs[11:15, 11:15]),
          [20.9996 -4.4261 4.0568 -0.8441 8.7149; -4.4261 24.6271 -4.7012 11.1628 -3.0641;
           4.0568 -4.7012 31.0621 -6.6603 19.0867; -0.8441 11.1628 -6.6603 38.6470 -8.8550;
           8.7149 -3.0641 19.0867 -8.8550 48.2930]; atol = 1e-2)
    x = solve_mme(m; zero = [10])                   # HTD 10 set to zero
    b, (a, pe) = solutions(m, x)
    a, pe = vec(a), vec(pe)
    check("HTD", b[1:10], [10.9783, 7.9951, 8.7031, 8.2806, 6.3813, 3.1893, 3.3099, 3.3897,
                           0.6751, 0]; atol = 1e-4)
    check("fixed regressions", b[11:15], [16.3082, -0.5227, -0.1245, 0.5355, -0.4195];
          atol = 2e-4)
    check("EBV (daily)", a, [-0.3300, -0.1604, 0.4904, 0.0043, -0.2449, -0.8367, 1.1477,
                             0.3786]; atol = 1e-4)
    check("EBV (305 days)", 305a, [-100.6476, -48.9242, 149.5718, 1.3203, -74.7065,
                                   -255.2063, 350.0481, 115.4757]; atol = 0.02)
    check("pe (daily)", pe, [-0.6156, -0.4151, -1.6853, 2.8089, -0.0928]; atol = 1e-4)
    # fixed lactation curve; the book's values from DIM 174 on (11.0407,
    # 10.9156, 11.1111, 11.2500, 10.8297) are not Φb̂₂ – e.g. Φ₁₇₄b̂₂ = 11.099
    v = Φ * b[11:15]
    show_table(DIM = dims, v = v; digits = 4)
    check("lactation curve v, DIM 4–140", v[1:5],
          [10.0835, 12.2001, 12.6254, 12.2077, 11.5679]; atol = 2e-4)

    # partition (Eqn 4.8): YD corrected for HTD, curve and pe
    yc = y - [X1 X2] * b - Q * pe
    nrec = [count(==(i), cow) for i in 1:8]
    yd = [nrec[i] > 0 ? sum(yc[cow.==i]) / nrec[i] : 0.0 for i in 1:8]
    p = ebv_partition(ped.sire, ped.dam, σ²e / σ²a, nrec, yd, a)
    check("YD of cows 6 and 8", yd[[6, 8]], [-0.9844, 0.3746]; atol = 1e-4)
    check("weights on YD of cows 6 and 8", p.n2[[6, 8]], [0.7882, 0.8815]; atol = 1e-4)
    check("weight on PA of cows 8 and 4", p.n1[[8, 4]], [0.1185, 0.1151]; atol = 1e-4)
    (; b, a, pe)
end
