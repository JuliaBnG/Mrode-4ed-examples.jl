# Example 6.4 – multivariate model with no environmental covariance:
# yearling weight on males and fat yield on females (relatives).

function ex_06_4()
    println("Example 6.4: different traits recorded on relatives")
    sire = [0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 3, 1, 3, 2, 2, 3]
    dam = [0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 7, 8, 0, 13, 15]
    ped = DataFrame(; sire, dam)
    calf = 9:17
    hys = [1, 2, 2, 1, 1, 2, 3, 2, 3]
    Y = [375.0 missing; 250 missing; 300 missing; 450 missing;
         missing 200; missing 160; missing 150; missing 250; missing 175]
    G0 = [43.0 18; 18 30]
    R0 = [77.0 0; 0 70]

    y, obs = mt_stack(Y)
    X, _ = incidence(hys)
    Z = incidence(calf, 17)
    Xb = mt_blocks(obs, [X, X])
    Xb[1] = Xb[1][:, 1:2]                   # no HYS 3 for yearling weight
    m = mme(y, reduce(hcat, Xb), [RandomEffect(mt_blocks(obs, [Z, Z]), ainv(ped), G0)];
            Rinv = mt_rinv(R0, obs))
    b, (a,) = solutions(m, solve_mme(m))
    show_table(animal = 1:17, weight = a[:, 1], fat = a[:, 2])
    check("HYS, yearling weight", b[1:2], [412.26, 276.21]; atol = 6e-3)
    check("HYS, fat", b[3:5], [194.03, 204.77, 161.66]; atol = 5e-3)
    # the book prints animal 6 as (−5.012, −2.098), a copy of animal 5; dam 6
    # has a son (11) with a high record, so her values are positive
    check("animals", a,
          [-3.365 1.258; -1.489 3.774; 4.237 -1.687; -6.940 -1.572; -5.012 -2.098;
           5.012 2.098; 2.137 3.561; -4.274 -7.123; -12.162 -3.091; -8.263 -1.260;
           5.836 3.776; 12.632 3.558; 1.523 5.971; -4.292 -11.527; -1.870 0.011;
           4.290 11.995; 2.684 1.663])
    (; b, a)
end
