# Test-day fat yields of Table 10.1, shared by Examples 10.1 and 10.2.

"""
    data_10() -> NamedTuple

Test-day fat yields of Table 10.1. `rec` holds the records in long format,
one row per test day (cow, DIM, HTD, TDY, and `dim`, the index of the DIM
in `dims`); they are typed as in the book, one column per cow, and stacked.
`ped` is the pedigree of Example 5.1 (IDs are row numbers), `Φ` the matrix
of normalized Legendre polynomials at the ten DIM (standardized over days
4–310).
"""
function data_10()
    ped =
        DataFrame(id = 1:8, sire = [0, 0, 0, 1, 3, 1, 3, 1], dam = [0, 0, 0, 2, 2, 5, 4, 7])
    dims = [4, 38, 72, 106, 140, 174, 208, 242, 276, 310]

    # Table 10.1: one column per cow, one row per DIM
    m3, m5 = fill(missing, 3), fill(missing, 5)
    tdy = DataFrame(
        "DIM" => dims,
        "4" => [17.0, 18.6, 24.0, 20.0, 20.0, 15.6, 16.0, 13.0, 8.2, 8.0],
        "5" => [23.0, 21.0, 18.0, 17.0, 16.2, 14.0, 14.2, 13.4, 11.8, 11.4],
        "6" => [[10.4, 12.3, 13.2, 11.6, 8.4]; m5],
        "7" => [[22.8, 22.4, 21.4, 18.8, 18.3, 16.2, 15.0]; m3],
        "8" => [22.2, 20.0, 21.0, 23.0, 16.8, 11.0, 13.0, 17.0, 13.0, 12.6],
    )
    htd = DataFrame(
        "DIM" => dims,
        "4" => 1:10,
        "5" => 1:10,
        "6" => [6:10; m5],
        "7" => [4:10; m3],
        "8" => 1:10,
    )

    # Long format: one row per (DIM, cow), without the empty cells
    long(df, v) = stack(df, Not(:DIM); variable_name = :cow, value_name = v)
    rec = dropmissing(innerjoin(long(htd, :htd), long(tdy, :tdy); on = [:DIM, :cow]))
    rec.cow = parse.(Int, rec.cow)
    rec.dim = Int.(indexin(rec.DIM, dims))
    sort!(rec, [:cow, :DIM])
    select!(rec, :cow, :DIM, :dim, :htd, :tdy)

    (; ped, rec, dims, Φ = legendre(dims, 5))
end
