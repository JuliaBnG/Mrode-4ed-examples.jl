# Example 15.1 – threshold (probit) sire model for calving ease in three
# ordered categories (Gianola and Foulley, 1983), with category
# probabilities by sire and a linear-model comparison.

function ex_15_1()
    println("Example 15.1: threshold model for calving ease")
    herd = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    sex = ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "M", "F", "M", "F", "M",
           "M", "F", "F", "M"]
    sire = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
    N = [1 0 0; 1 0 0; 1 0 0; 0 1 0; 1 0 1; 3 0 0; 1 1 0; 0 1 0; 1 0 0; 2 0 0; 1 0 0; 0 0 1;
         1 0 1; 1 0 0; 0 1 0; 0 0 1; 0 1 0; 1 0 0; 2 0 0; 2 0 0]
    sped = DataFrame(sire = [0, 0, 1, 3], dam = [0, 0, 0, 0])
    Ai = Matrix(ainv(sped))
    X = [first(incidence(herd)) first(incidence(sex; levels = ["M", "F"]))]
    Z, _ = incidence(sire)
    # starting thresholds as in the book (from tables of the normal curve)
    r = threshold_model(N, X, Z, Ai, 19; t0 = [0.468, 1.080], zero = [1, 3])
    check("after iteration 1: t, b, u", r.history[:, 1],
          [0.441008, 1.044792, 0, 0.286869, 0, -0.358323, -0.041528, 0.057853, 0.039850,
           -0.065178]; atol = 1e-5)
    check("after iteration 2", r.history[:, 2],
          [0.4375, 1.0661, 0, 0.2763, 0, -0.3577, -0.0431, 0.0586, 0.0410, -0.0653];
          atol = 1e-4)
    println("  converged after ", r.iterations, " iterations")
    keep = setdiff(1:10, [3, 5])
    C = zeros(10, 10)
    C[keep, keep] = inv(r.lhs[keep, keep])
    se = sqrt.(diag(C))
    show_table(effect = ["t1", "t2", "herd 1", "herd 2", "male", "female", "sire " .* string.(1:4)...],
               solution = [r.t; r.b; r.u], se = se; digits = 4)
    check("final solutions", [r.t; r.b; r.u],
          [0.4378, 1.0675, 0, 0.2774, 0, -0.3590, -0.0434, 0.0592, 0.0412, -0.0660];
          atol = 1e-4)
    check("standard errors", se[[1, 2, 4, 6, 7, 8, 9, 10]],
          [0.44, 0.47, 0.49, 0.48, 0.22, 0.21, 0.22, 0.22]; atol = 0.005)

    # probabilities for female calves in herd–year 1, by sire
    η = r.b[1] + r.b[4] .+ r.u
    P = category_probabilities(r.t, η)
    # (the book's row for sire 4 is off by 0.003: Φ(0.4378 + 0.3590 + 0.0660)
    # = 0.806, not 0.803)
    check("P(category | herd 1, female, sire)", P,
          [0.800 0.129 0.071; 0.770 0.145 0.086; 0.775 0.142 0.083; 0.803 0.129 0.068];
          atol = 3e-3)
    # averaged over the four herd–year × sex subclasses
    Pbar = sum(category_probabilities(r.t, r.b[h] + r.b[2+s] .+ r.u) for h in 1:2, s in 1:2) ./ 4
    check("P(category | sire), all subclasses", Pbar,
          [0.695 0.175 0.131; 0.659 0.188 0.153; 0.665 0.186 0.149; 0.702 0.172 0.126];
          atol = 2e-3)

    # Linear sire model on the category scores, α = 19. The book's linear
    # solutions (herd 2 −0.0819, male 1.3457, female 1.6073, …) have the
    # opposite signs to the raw category means and could not be reproduced
    # from Table 15.2; the solutions from these data are printed instead.
    rec = [(j, k) for j in axes(N, 1) for k in 1:3 for _ in 1:N[j, k]]
    j, y = first.(rec), Float64.(last.(rec))
    m = mme(y, X[j, :], [RandomEffect(Z[j, :], Ai, 1 / 19)]; σ²e = 1.0)
    x = solve_mme(m; zero = [1])
    show_table(effect = ["herd 2", "male", "female", "sire " .* string.(1:4)...],
               linear = x[2:8]; digits = 4)
    r
end
