# Test-day fat yields of Table 10.1, shared by Examples 10.1 and 10.2.

"""
    data_10() -> NamedTuple

Records in long format (cow, DIM, HTD, TDY), the pedigree of
Example 5.1 and the matrix Φ of normalized Legendre polynomials at
the ten DIM (standardized over days 4–310).
"""
function data_10()
    ped = DataFrame(sire = [0, 0, 0, 1, 3, 1, 3, 1], dam = [0, 0, 0, 2, 2, 5, 4, 7])
    dims = [4, 38, 72, 106, 140, 174, 208, 242, 276, 310]
    tdy = Dict(
        4 => [17.0, 18.6, 24.0, 20.0, 20.0, 15.6, 16.0, 13.0, 8.2, 8.0],
        5 => [23.0, 21.0, 18.0, 17.0, 16.2, 14.0, 14.2, 13.4, 11.8, 11.4],
        6 => [10.4, 12.3, 13.2, 11.6, 8.4],
        7 => [22.8, 22.4, 21.4, 18.8, 18.3, 16.2, 15.0],
        8 => [22.2, 20.0, 21.0, 23.0, 16.8, 11.0, 13.0, 17.0, 13.0, 12.6],
    )
    firsthtd = Dict(4 => 1, 5 => 1, 6 => 6, 7 => 4, 8 => 1)
    cow, dim, htd, y = Int[], Int[], Int[], Float64[]
    for c = 4:8, (j, v) in enumerate(tdy[c])
        push!(cow, c)
        push!(dim, j)
        push!(htd, firsthtd[c] + j - 1)
        push!(y, v)
    end
    (; ped, dims, cow, dim, htd, y, Φ = legendre(dims, 5))
end
