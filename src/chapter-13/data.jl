# Data of Examples 13.1 (pedigree) and 13.3–13.5 (pigs 1–15 with 20 SNPs).

"""
    data_13_1() -> NamedTuple

Pedigree, management pens, and weaning weights of pigs 1–12 (records on pigs 5–12)
used in Examples 13.1 and 13.2 for pedigree additive and dominance models.
"""
data_13_1() = (
    ped = DataFrame(
        sire = [0, 0, 0, 0, 1, 3, 6, 0, 3, 3, 6, 6],
        dam = [0, 0, 0, 0, 2, 4, 5, 5, 8, 8, 8, 8],
    ),
    pig = 5:12,
    pen = [1, 1, 1, 1, 2, 2, 2, 2],
    ww = [17.0, 20.0, 18.0, 13.5, 20.0, 15.0, 25.0, 19.5],
    σ²a = 90.0,
    σ²d = 80.0,
    σ²e = 120.0,
)

"""
    data_13_3() -> NamedTuple

Genotypes (15 pigs, 20 SNPs), management pens, and weaning weights (records on pigs 5–15)
used in Examples 13.3–13.5 for genomic additive, dominance, and epistatic models.
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
    (;
        M,
        pig = 5:15,
        pen = [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2],
        ww = [17.0, 20.0, 18.0, 13.5, 20.0, 15.0, 25.0, 19.5, 22.5, 16.0, 24.5],
        σ²a = 90.0,
        σ²d = 80.0,
        σ²e = 120.0,
    )
end
