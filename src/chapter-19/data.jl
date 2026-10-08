# Example 4.1 data and MME, shared by the Chapter 19 examples.

function data_19()
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 4, 3], dam = [0, 0, 0, 0, 2, 2, 5, 6])
    calf = [4, 5, 6, 7, 8]
    sex = [1, 2, 2, 1, 1]
    y = [4.5, 2.9, 3.9, 3.5, 5.0]
    m = mme(y, first(incidence(sex)), [RandomEffect(incidence(calf, 8), ainv(ped), 20.0)];
            σ²e = 40.0)
    # Mrode's starting values: sex means, records − sex mean, ancestors 0
    x0 = [4.333, 3.400, 0, 0, 0, 0.167, -0.500, 0.500, -0.833, 0.667]
    (; ped, calf, sex, y, m, x0, α = 2.0)
end
