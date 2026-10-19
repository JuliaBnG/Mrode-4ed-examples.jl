# Data of Examples 13.1 (pedigree) and 13.3–13.5 (pigs 1–15 with 20 SNPs).

"""
    data_13_1() -> NamedTuple

Pedigree, management pens, and weaning weights of pigs 1–12 (records on pigs 5–12)
used in Examples 13.1 and 13.2 for pedigree additive and dominance models. `ped` holds
all of it, one row per pig (IDs are row numbers; pigs 1–4 have no records).
"""
function data_13_1()
    no4 = fill(missing, 4)
    ped = DataFrame(
        pig = 1:12,
        sire = [0, 0, 0, 0, 1, 3, 6, 0, 3, 3, 6, 6],
        dam = [0, 0, 0, 0, 2, 4, 5, 5, 8, 8, 8, 8],
        pen = [no4; [1, 1, 1, 1, 2, 2, 2, 2]],
        ww = [no4; [17.0, 20.0, 18.0, 13.5, 20.0, 15.0, 25.0, 19.5]],
    )
    @assert ped.pig == 1:nrow(ped)
    (; ped, σ²a = 90.0, σ²d = 80.0, σ²e = 120.0)
end

"""
    data_13_3() -> NamedTuple

Genotypes (15 pigs, 20 SNPs), management pens, and weaning weights (records on pigs 5–15)
used in Examples 13.3–13.5 for genomic additive, dominance, and epistatic models.
`pigs` has one row per pig (in the row order of `M`; pigs 1–4 have no records).
"""
function data_13_3()
    M = [
        2 2 0 0 1 1 1 0 1 1 0 0 1 0 1 2 0 1 0 1
        1 1 1 1 2 0 1 0 2 2 0 1 0 1 1 1 0 0 0 0
        1 2 1 0 2 0 1 1 2 2 0 1 2 0 2 2 0 0 1 0
        1 1 0 0 2 0 2 2 2 2 1 1 0 1 0 2 1 0 1 0
        1 1 1 1 1 0 2 0 1 2 0 0 0 1 1 2 0 0 0 1
        2 1 0 0 2 0 1 1 2 2 0 1 1 1 1 2 0 0 2 0
        2 1 0 0 1 0 2 1 1 2 0 1 0 1 1 2 0 0 1 1
        2 2 0 0 1 0 2 1 1 2 0 0 1 0 2 1 0 0 0 1
        2 2 0 0 1 0 2 2 2 2 0 1 2 0 2 1 0 0 0 0
        1 2 0 0 2 0 2 2 2 2 0 1 2 0 2 1 0 0 0 0
        2 1 0 0 1 0 1 0 1 2 0 0 1 0 2 1 0 0 1 0
        2 1 0 0 1 0 2 1 1 2 0 1 0 1 1 2 0 0 1 1
        2 1 0 0 2 0 2 2 2 2 0 1 1 1 1 1 0 0 1 0
        2 1 0 0 1 0 2 2 2 2 0 1 1 0 2 1 0 0 0 1
        2 1 0 0 0 0 2 0 1 2 0 0 0 0 2 1 0 0 0 1
    ]
    no4 = fill(missing, 4)
    pigs = DataFrame(
        pig = 1:15,
        pen = [no4; [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2]],
        ww = [no4; [17.0, 20.0, 18.0, 13.5, 20.0, 15.0, 25.0, 19.5, 22.5, 16.0, 24.5]],
    )
    (;
        M,
        pigs,
        σ²a = 90.0,
        σ²d = 80.0,
        σ²e = 120.0,
    )
end
