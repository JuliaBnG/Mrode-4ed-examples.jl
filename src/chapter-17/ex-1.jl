# Sections 17.3 and 17.5 – sire variance by Henderson's method 3 and by
# fitting the canonical sums of squares (OLS and iterated GLS).

function ex_17_1()
    println("Sections 17.3 and 17.5: variance components of a sire model")
    sire = [2, 1, 3, 2]
    y = [2.9, 4.0, 3.5, 3.5]
    X = ones(4, 1)
    Z = Matrix(first(incidence(sire)))
    h = henderson3(y, X, Z)
    check("F, S, R", [h.F, h.S, h.R], [48.3025, 0.4275, 0.1800]; atol = 1e-10)
    check("Z'SZ", h.ZSZ, [0.75 -0.5 -0.25; -0.5 1 -0.5; -0.25 -0.5 0.75]; atol = 1e-12)
    check("σ²e, σ²s (method 3)", [h.σ²e, h.σ²s], [0.18, 0.027])

    c = sire_contrasts(y, X, Z, Matrix(1.0I, 3, 3))
    check("W", c.w, [1.5, 1.0]; atol = 1e-12)
    check("|Q|", abs.(c.Q'), [0.3333 0.6667 0.3333; 0.7071 0 0.7071]; atol = 1e-4)
    check("contrast sums of squares", c.ss, [0.3025, 0.1250]; atol = 1e-12)
    ms = [c.ss; c.R / c.dfR]
    df = [1, 1, c.dfR]
    C = [ones(3) [c.w; 0]]
    # The book reports 0.143 and 0.079 for the unweighted fit; ordinary
    # least squares on these three sums of squares gives 0.151 and 0.062.
    θ, _ = fit_mean_squares(ms, df, C; weighted = false)
    @printf("  OLS fit: σ²e = %.3f, σ²s = %.3f (book 0.143, 0.079)\n", θ...)
    θw, V = fit_mean_squares(ms, df, C)
    check("iterated GLS fit: σ²e, σ²s", θw, [0.163, 0.047])
    # the book's "estimated variances" 0.216 and 0.234 are not reproduced;
    # the sampling variances here are (0.051, 0.062), SEs (0.226, 0.250)
    @printf("  sampling variances %.3f, %.3f; standard errors %.3f, %.3f\n", diag(V)...,
            sqrt.(diag(V))...)
    (; h, c, θ, θw)
end
