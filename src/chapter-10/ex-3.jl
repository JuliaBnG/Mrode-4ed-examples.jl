"""
    ex_10_3() -> NamedTuple

Example 10.3: Covariance functions and linear spline models
(Mrode & Pocrnic, 2023, Sections 10.4 and 10.3.5; Kirkpatrick et al., 1990; Meyer, 2005).

# Model Formulation and Theory
Covariance functions provide a continuous representation of genetic (co)variance across age
or time. Given a discrete covariance matrix ``G`` at ages ``t_1, t_2, \\dots, t_p`` and
normalized Legendre polynomials ``\\Phi``:
```math
G = \\Phi C \\Phi' \\implies \\hat{C} = \\Phi^{-1} G (\\Phi')^{-1}
```
where:
- ``C`` is the coefficient matrix of the covariance function.
-  Covariance between any two ages ``t_i`` and ``t_j`` is given by
  ``\\phi(t_i)' C \\phi(t_j)``.
-  Reduced-order approximation (Section 10.4.1): Fitting a ``k \\times k`` coefficient
  matrix (here ``k = 2``) to an original ``3 \\times 3`` matrix via Weighted Least Squares
  (WLS) using the asymptotic covariance matrix of sample (co)variances ``V``, evaluated by a
  ``\\chi^2`` goodness-of-fit test.
-  Linear spline model (Section 10.3.5): Continuous piecewise linear functions defined by
  knot points.

# Returns
A `NamedTuple` with fields:
- `f`: Full-order covariance function fit (`f.C`, `f.T`, `f.Φ`).
- `r`: Reduced-order covariance function fit (`r.C`, `r.χ²`, `r.df`).
"""
function ex_10_3()
    println("Section 10.4: covariance functions for body weight at 90, 160, 240")

    # 1. Discrete covariance matrix and recording ages
    G = [132.3 127.0 136.6; 127.0 172.8 200.8; 136.6 200.8 288.0]
    ages = [90, 160, 240]

    # 2. Full-order covariance function fit (Eqn 10.9)
    f = covariance_function(G, ages)
    check(
        "Φ",
        f.Φ,
        [0.7071 -1.2247 1.5811; 0.7071 -0.0816 -0.7801; 0.7071 1.2247 1.5811];
        atol = 1e-4,
    )
    check("Ĉ, second row", f.C[2, :], [45.2787, 24.5185, -0.1475]; atol = 3e-3)
    check(
        "T",
        f.T,
        [177.99 39.35 -11.52; 39.35 36.78 -0.43; -11.52 -0.43 18.43];
        atol = 6e-3,
    )

    # 3. Predict variances and covariances at arbitrary ages (e.g. 90 and 200 days)
    ϕ90, ϕ200 = eachrow(legendre([90, 200], 3; tmin = 90, tmax = 240))
    check(
        "var(90), var(200), cov(90, 200)",
        [ϕ90' * f.C * ϕ90, ϕ200' * f.C * ϕ200, ϕ90' * f.C * ϕ200],
        [132.30, 218.50, 129.71];
        atol = 0.01,
    )

    # 4. Reduced-order fit by weighted least squares (k = 2) and χ² test (Section 10.4.1)
    V = [
        3450.0 2256.4 2184.6 1480.3 1434.7 1390.9;
        2256.4 2959.6 2430.9 2903.5 2530.2 2180.1;
        2184.6 2430.9 3889.7 2249.1 3181.6 4051.5;
        1480.3 2903.5 2249.1 5711.4 4410.0 3417.5;
        1434.7 2530.2 3181.6 4410.0 5818.8 6354.3;
        1390.9 2180.1 4051.5 3417.5 6354.3 11835.0
    ]
    r = covariance_function(G, ages; k = 2, V)
    check("reduced fit Č", r.C, [341.8512 45.0421; 45.0421 24.5405]; atol = 0.01)
    check("χ² (3 df)", [r.χ², r.df], [0.2231, 3]; atol = 1e-3)

    # 5. Linear spline covariables with knots at days 4, 106, 208, 310 (Section 10.3.5)
    # Note: Mrode's table puts the larger weight on the farther knot; standard linear
    # interpolation is computed here.
    F = linear_spline([4, 38, 72, 106, 140, 174, 208, 242, 276, 310], [4, 106, 208, 310])
    show_table(
        DIM = [4, 38, 72, 106, 140, 174, 208, 242, 276, 310],
        F0 = F[:, 1],
        F1 = F[:, 2],
        F2 = F[:, 3],
        F3 = F[:, 4],
    )

    (; f, r)
end
