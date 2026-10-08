# Helpers shared by the Chapter 14 examples.

"""
    nzinv(A; tol = 1e-12) -> (Ainv, null)

Generalized inverse of a partial relationship matrix: computes the inverse of
its non-zero submatrix, keeping zero rows and columns for animals with zero contribution
(`null`, which must be constrained to zero in the MME).
"""
function nzinv(A::AbstractMatrix; tol = 1e-12)
    nz = findall(i -> any(x -> abs(x) > tol, view(A, :, i)), axes(A, 2))
    K = zeros(size(A))
    K[nz, nz] = inv(Symmetric(A[nz, nz]))
    K, setdiff(axes(A, 1), nz)
end

"""
    data_14_1() -> NamedTuple

Pedigree (11 animals), founder breed fractions, management herds, phenotypes,
and variance components for Examples 14.1 and 14.2 (García-Cortés & Toro, 2006).
"""
function data_14_1()
    ped = DataFrame(
        sire = [0, 0, 0, 0, 1, 3, 3, 5, 7, 9, 5],
        dam = [0, 0, 0, 0, 2, 2, 4, 6, 6, 8, 8],
    )
    founder = zeros(11, 2)
    founder[1:2, 1] .= 1
    founder[3:4, 2] .= 1
    (;
        ped,
        F = breed_composition(ped, founder),
        herd = [2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1],
        y = collect(11.0:21.0),
        σ² = [1.0, 2.0, 0.5],
        σ²e = 4.0,
    )
end

"""
    data_14_3() -> NamedTuple

Pedigree (12 animals: purebred breed 1, purebred breed 2, and crossbred F1 progeny),
herd allocations, records, and genetic covariance matrices for Examples 14.3 and 14.4
(Christensen et al., 2014).
"""
function data_14_3()
    ped = DataFrame(
        sire = [0, 0, 1, 1, 0, 0, 5, 5, 1, 1, 5, 5],
        dam = [0, 0, 2, 3, 0, 0, 6, 7, 6, 7, 2, 3],
    )
    group = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]        # breed 1, breed 2, crossbred
    (;
        ped,
        group,
        herd = [2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2],
        y = collect(11.0:22.0),
        S1 = [1.0 0.92; 0.92 1.5],
        S2 = [2.0 1.385; 1.385 1.5],
        σ²e = 4.0,
        σ²x = 4.0 + 1.5 - 0.25 * (1.0 + 2.0),
    )           # crossbred residual, 4.75
end
