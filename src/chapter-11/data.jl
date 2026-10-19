# Data of Example 11.1 (ten SNPs of 14 bulls), shared by Chapter 11 and 12.

"""
    data_11() -> NamedTuple

Pedigree of animals 1–26 (1–12 are ungenotyped ancestors), genotypes
`M` (14 × 10) of bulls 13–26, fat DYD and EDC. Bulls 13–20 are the
reference population, 21–26 the selection candidates.

`ped` has one row per animal (IDs are row numbers) and `bulls` one row per
genotyped bull (`id`, `set` = "reference" or "candidate", `dyd`, `edc`), in
the row order of `M`. The other fields are the same data as vectors and
index ranges into `bulls`.
"""
function data_11()
    sire = vcat(zeros(Int, 14), [13, 15, 15, 14, 14, 14, 1, 14, 14, 14, 14, 14])
    dam = vcat(zeros(Int, 14), [4, 2, 5, 6, 9, 9, 3, 8, 11, 10, 7, 12])
    M = [
        2 0 1 1 0 0 0 2 1 2
        1 0 0 0 0 2 0 2 1 0
        1 1 2 1 1 0 0 2 1 2
        0 0 2 1 0 1 0 2 2 1
        0 1 1 2 0 0 0 2 1 2
        1 1 0 1 0 2 0 2 2 1
        0 0 1 1 0 2 0 2 2 0
        0 1 1 0 0 1 0 2 2 0
        2 0 0 0 0 1 2 2 1 2
        0 0 0 1 1 2 0 2 0 0
        0 1 1 0 0 1 0 2 2 1
        1 0 0 0 1 1 0 2 0 0
        0 0 0 1 1 2 0 2 1 0
        1 0 1 1 0 2 0 1 0 0
    ]
    dyd = [9.0, 13.4, 12.7, 15.4, 5.9, 7.7, 10.2, 4.8, 7.6, 8.8, 9.8, 9.2, 11.5, 13.3]
    edc = [558, 722, 300, 73, 52, 87, 64, 103, 13, 125, 93, 66, 75, 33]
    id = 13:26
    bulls = DataFrame(
        id = id,
        set = [fill("reference", 8); fill("candidate", 6)],
        dyd = dyd,
        edc = edc,
    )
    (;
        ped = DataFrame(id = 1:26, sire = sire, dam = dam),
        bulls,
        M,
        dyd,
        edc,
        id,
        ref = 1:8,
        cand = 9:14,
        σ²a = 35.241,
        σ²e = 245.0,
    )
end
