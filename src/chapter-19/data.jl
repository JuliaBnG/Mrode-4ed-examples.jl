"""
    data_19() -> NamedTuple

Construct the dataset and univariate animal model MME from Example 4.1
for the iterative solving methods in Chapter 19.

# Returns
A `NamedTuple` with fields:
- `ped`: The data of Example 4.1, one row per animal (pedigree, sex coded 1/2, record).
- `calf`, `sex`, `y`: Phenotypic observation vectors.
- `m`: The `MME` system with ``\\sigma_e^2 = 40.0, \\sigma_a^2 = 20.0`` (``\\alpha = 2.0``).
- `x0`: Mrode's initial starting vector (sex means for fixed effects,
  deviations from sex mean for recorded animals, 0 for ancestors), computed from the
  records with `groupby`.
- `α`: Variance ratio ``\\alpha = \\sigma_e^2 / \\sigma_a^2 = 2.0``.
"""
function data_19()
    no3 = fill(missing, 3)
    ped = DataFrame(
        calf = 1:8,
        sex = [no3; [1, 2, 2, 1, 1]],
        sire = [0, 0, 0, 1, 3, 1, 4, 3],
        dam = [0, 0, 0, 0, 2, 2, 5, 6],
        y = [no3; [4.5, 2.9, 3.9, 3.5, 5.0]],
    )
    rec = dropmissing(ped, :y)
    calf, sex, y = rec.calf, Int.(rec.sex), rec.y
    m = mme(
        y,
        first(incidence(sex)),
        [RandomEffect(incidence(calf, 8), ainv(ped), 20.0)];
        σ²e = 40.0,
    )
    # Mrode's starting values: sex means, records − sex mean, ancestors 0
    dev = transform(groupby(rec, :sex), :y => (v -> v .- mean(v)) => :dev)
    smean = combine(groupby(rec, :sex), :y => mean => :mean)
    a0 = zeros(nrow(ped))
    a0[dev.calf] = dev.dev
    x0 = [sort(smean, :sex).mean; a0]
    (; ped, calf, sex, y, m, x0, α = 2.0)
end
