# Section 17.7 – REML for the animal model by average information
# (Table 17.3) and by the EM-type updates of Eqns 17.5–17.6.

function ex_17_2()
    println("Section 17.7: REML for an animal model")
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = 4:8
    sex = ["M", "F", "F", "M", "M"]
    y = [2.6, 0.1, 1.0, 3.0, 1.0]
    X, _ = incidence(sex; levels = ["M", "F"])
    a = RandomEffect(incidence(calf, 8), ainv(ped), 0.2)

    # first iterate in detail
    m = mme(y, X, [a]; σ²e = 0.4)
    x = solve_mme(m)
    # the book prints â₄ = −0.254; its own residual of calf 4 (0.2022) needs +0.254
    check("MME solutions at σ²e = 0.4, σ²a = 0.2", x,
          [2.144, 0.602, 0.117, -0.025, -0.222, 0.254, -0.135, 0.032, 0.219, -0.305])
    check("residuals", y - [X a.Z] * x, [0.2022, -0.3661, 0.3661, 0.6374, -0.8395];
          atol = 1e-4)
    check("C²²σ²e, first row", lhs_inverse(m)[3, 3:10] .* 0.4,
          [0.1884, 0.0028, 0.0131, 0.0878, 0.0180, 0.0883, 0.0554, 0.0537]; atol = 1e-4)

    r = reml(y, X, [a]; σ²e = 0.4, method = :ai)
    show_table(iterate = 1:length(r.logLs), σ²e = r.history[1, 1:end-1],
               σ²a = r.history[2, 1:end-1], logL = r.logLs; digits = 4)
    check("AI-REML iterates σ²e", r.history[1, 1:6],
          [0.4000, 0.4838, 0.4910, 0.4839, 0.4835, 0.4835]; atol = 1e-4)
    check("AI-REML iterates σ²a", r.history[2, 1:6],
          [0.2000, 0.3695, 0.5126, 0.5500, 0.5514, 0.5514]; atol = 1e-4)
    check("log-likelihoods", r.logLs[1:6],
          [-2.3852, -2.2021, -2.1821, -2.1817, -2.1817, -2.1817]; atol = 1e-4)
    check("AI⁻¹ at convergence", r.AIinv, [2.4436 -3.2532; -3.2532 5.3481]; atol = 1e-3)
    @printf("  standard errors: σ²e %.3f, σ²a %.3f\n", sqrt.(diag(r.AIinv))...)

    e1 = reml(y, X, [a]; σ²e = 0.4, method = :em, maxiter = 1)
    check("first EM iterate (σ²e, σ²a)", e1.history[:, 2], [0.6426, 0.2125])
    em = reml(y, X, [a]; σ²e = 0.4, method = :em, maxiter = 1000)
    check("EM after 1000 iterates (σ²e, σ²a, logL)", [em.history[:, end]; em.logLs[end]],
          [0.4842, 0.5504, -2.1817])
    (; r, em)
end
