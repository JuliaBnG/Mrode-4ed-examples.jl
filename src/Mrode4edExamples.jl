"""
    Mrode4edExamples

Julia solutions to the worked examples of Mrode and Pocrnic (2023),
*Linear Models for the Prediction of the Genetic Merit of Animals*, 4th
edition, built on [Breeding.jl](https://github.com/JuliaBnG/Breeding.jl)
and [RelationshipMatrices.jl](https://github.com/JuliaBnG/RelationshipMatrices.jl).

Every example is a function `ex_CC_N()` (chapter `CC`, script `N`) that
prints its results, compares them with the values printed in the book
and returns them. Use [`list_examples`](@ref) to see the catalogue and
[`run_examples`](@ref) to run (and check) several at once.
"""
module Mrode4edExamples

using Breeding
using DataFrames
using Distributions
using LinearAlgebra
using Printf
using Random
using RelationshipMatrices
using SparseArrays
using Statistics

include("common.jl")

include("chapter-04/ex-1.jl")
include("chapter-04/ex-2.jl")
include("chapter-04/ex-3.jl")
include("chapter-04/ex-4.jl")
include("chapter-05/ex-1.jl")
include("chapter-05/ex-2.jl")
include("chapter-06/ex-1.jl")
include("chapter-06/ex-2.jl")
include("chapter-06/ex-3.jl")
include("chapter-06/ex-4.jl")
include("chapter-06/ex-5.jl")
include("chapter-08/ex-1.jl")
include("chapter-08/ex-2.jl")
include("chapter-09/ex-1.jl")
include("chapter-10/data.jl")
include("chapter-10/ex-1.jl")
include("chapter-10/ex-2.jl")
include("chapter-10/ex-3.jl")
include("chapter-11/data.jl")
include("chapter-11/ex-1.jl")
include("chapter-11/ex-2.jl")
include("chapter-11/ex-3.jl")
include("chapter-11/ex-6.jl")
include("chapter-11/ex-7.jl")
include("chapter-11/ex-8.jl")
include("chapter-11/ex-9.jl")
include("chapter-11/ex-13.jl")
include("chapter-12/ex-1.jl")
include("chapter-12/ex-2.jl")
include("chapter-13/data.jl")
include("chapter-13/ex-1.jl")
include("chapter-13/ex-2.jl")
include("chapter-13/ex-3.jl")
include("chapter-13/ex-5.jl")
include("chapter-14/common14.jl")
include("chapter-14/ex-1.jl")
include("chapter-14/ex-3.jl")
include("chapter-15/ex-1.jl")
include("chapter-15/ex-2.jl")
include("chapter-17/ex-1.jl")
include("chapter-17/ex-2.jl")
include("chapter-18/ex-1.jl")
include("chapter-18/ex-2.jl")
include("chapter-19/data.jl")
include("chapter-19/ex-1.jl")
include("chapter-19/ex-3.jl")
include("chapter-19/ex-5.jl")

"""
    EXAMPLES

Catalogue of the examples: function name, chapter, source file (under
`src/`) and title.
"""
const EXAMPLES = [
    (name = :ex_04_1, chapter = 4, file = "chapter-04/ex-1.jl",
     title = "Example 4.1 – univariate animal model, accuracy of evaluations (4.3.3) and partition of breeding values into PA, YD, PC and DYD (4.3.1–4.3.2)."),
    (name = :ex_04_2, chapter = 4, file = "chapter-04/ex-2.jl",
     title = "Example 4.2 – sire model."),
    (name = :ex_04_3, chapter = 4, file = "chapter-04/ex-3.jl",
     title = "Example 4.3 – reduced animal model, with back-solving for non-parents."),
    (name = :ex_04_4, chapter = 4, file = "chapter-04/ex-4.jl",
     title = "Example 4.4 – animal model with unknown parent groups (QP transformation)."),
    (name = :ex_05_1, chapter = 5, file = "chapter-05/ex-1.jl",
     title = "Example 5.1 – repeatability model, and daughter yield deviation (5.2.3)."),
    (name = :ex_05_2, chapter = 5, file = "chapter-05/ex-2.jl",
     title = "Example 5.2 – animal model with common environmental (full-sib) effects."),
    (name = :ex_06_1, chapter = 6, file = "chapter-06/ex-1.jl",
     title = "Example 6.1 – multivariate animal model, equal design matrices and no missing records; partition of evaluations (6.2.3) and reliabilities (6.2.4)."),
    (name = :ex_06_2, chapter = 6, file = "chapter-06/ex-2.jl",
     title = "Example 6.2 – multivariate animal model, equal design matrices with missing records."),
    (name = :ex_06_3, chapter = 6, file = "chapter-06/ex-3.jl",
     title = "Example 6.3 – multivariate model with unequal design matrices (a different HYS effect for each lactation), and DYD from a multivariate model (6.4.2)."),
    (name = :ex_06_4, chapter = 6, file = "chapter-06/ex-4.jl",
     title = "Example 6.4 – multivariate model with no environmental covariance: yearling weight on males and fat yield on females (relatives)."),
    (name = :ex_06_5, chapter = 6, file = "chapter-06/ex-5.jl",
     title = "Example 6.5 – multi-trait across-country evaluation (MACE) of bulls with a sire–maternal-grandsire model and phantom groups, and the partition of MACE proofs (Eqns 6.19–6.21)."),
    (name = :ex_08_1, chapter = 8, file = "chapter-08/ex-1.jl",
     title = "Example 8.1 – animal model for a maternal trait: direct and maternal genetic effects (correlated) and maternal permanent environment."),
    (name = :ex_08_2, chapter = 8, file = "chapter-08/ex-2.jl",
     title = "Example 8.2 – reduced animal model with maternal effects, and back-solving direct and maternal effects of non-parents (Eqns 8.8, 8.10)."),
    (name = :ex_09_1, chapter = 9, file = "chapter-09/ex-1.jl",
     title = "Example 9.1 – animal model with social interaction (associative) effects: random group effect (Eqn 9.6), no associative effects, the correlated-residual model (Eqn 9.7, Section 9.4), and the partition of DBV and SBV (Eqn 9.9)."),
    (name = :ex_10_1, chapter = 10, file = "chapter-10/ex-1.jl",
     title = "Example 10.1 – fixed regression test-day model: one Legendre lactation curve of order 4, animal and permanent environment effects."),
    (name = :ex_10_2, chapter = 10, file = "chapter-10/ex-2.jl",
     title = "Example 10.2 – random regression test-day model: Legendre polynomials of order 2 for animal and pe effects; 305-day values, partition of random regressions (10.3.2) and reliabilities (10.3.4)."),
    (name = :ex_10_3, chapter = 10, file = "chapter-10/ex-3.jl",
     title = "Section 10.4 – covariance functions: full-order fit (Eqn 10.9) and a reduced-order fit by weighted least squares (10.4.1), plus the linear spline covariables of 10.3.5."),
    (name = :ex_11_1, chapter = 11, file = "chapter-11/ex-1.jl",
     title = "Example 11.1 – SNPs fitted as fixed effects together with a random polygenic (animal) effect, without and with EDC weights."),
    (name = :ex_11_2, chapter = 11, file = "chapter-11/ex-2.jl",
     title = "Example 11.2 – SNP-BLUP (random SNP effects), unweighted and with EDC weights."),
    (name = :ex_11_3, chapter = 11, file = "chapter-11/ex-3.jl",
     title = "Examples 11.3–11.5 – GBLUP, SNP effects back-solved from GBLUP, the selection-index form, and cross-validation/reliability (Section 11.9)."),
    (name = :ex_11_6, chapter = 11, file = "chapter-11/ex-6.jl",
     title = "Example 11.6 – base-population allele frequencies and gene content of ungenotyped ancestors by the linear method (Gengler et al., 2007)."),
    (name = :ex_11_7, chapter = 11, file = "chapter-11/ex-7.jl",
     title = "Example 11.7 – SNP-BLUP and GBLUP with a residual polygenic effect (10 % of the additive genetic variance)."),
    (name = :ex_11_8, chapter = 11, file = "chapter-11/ex-8.jl",
     title = "Example 11.8 – haplotype model: pseudo-SNPs from phased haplotypes and GBLUP with the haplotype-based relationship matrix."),
    (name = :ex_11_9, chapter = 11, file = "chapter-11/ex-9.jl",
     title = "Examples 11.9–11.12 – BayesA, BayesB, BayesC and BayesCπ with residual updating (uncentred genotypes, flat priors on the mean and σ²e).  These are MCMC estimates: the book's first-iteration values depend on its random numbers, and its posterior means (7000 samples) carry Monte Carlo error. Here the deterministic start is checked exactly and the posterior means from a longer chain are compared with tolerances that reflect that error."),
    (name = :ex_11_13, chapter = 11, file = "chapter-11/ex-13.jl",
     title = "Example 11.13 – multivariate GBLUP (WWG and PWG of the beef calves)."),
    (name = :ex_12_1, chapter = 12, file = "chapter-12/ex-1.jl",
     title = "Example 12.1 – single-step GBLUP: G blended with A₂₂, tuned to the pedigree base, and H⁻¹ = A⁻¹ + [0 0; 0 Gt⁻¹ − A₂₂⁻¹]."),
    (name = :ex_12_2, chapter = 12, file = "chapter-12/ex-2.jl",
     title = "Example 12.2 – inverse of G by the APY algorithm (core: bulls 15, 20, 22 and 26) and GBLUP with it."),
    (name = :ex_13_1, chapter = 13, file = "chapter-13/ex-1.jl",
     title = "Example 13.1 – animal model with additive and dominance effects from the pedigree (D by Cockerham's rule, Eqn 13.1)."),
    (name = :ex_13_2, chapter = 13, file = "chapter-13/ex-2.jl",
     title = "Example 13.2 – solving directly for total genetic merit g = a + d with var(g) = Aσ²a + Dσ²d, and recovering a and d from g."),
    (name = :ex_13_3, chapter = 13, file = "chapter-13/ex-3.jl",
     title = "Examples 13.3 and 13.4 – genomic additive and dominance relationship matrices (Vitezica et al., 2013), without and with genomic inbreeding as a covariate for inbreeding depression."),
    (name = :ex_13_5, chapter = 13, file = "chapter-13/ex-5.jl",
     title = "Example 13.5 – genomic model with additive, dominance and additive × additive epistatic effects."),
    (name = :ex_14_1, chapter = 14, file = "chapter-14/ex-1.jl",
     title = "Examples 14.1 and 14.2 – multibreed animal model with breed-specific partial relationship matrices (García-Cortés and Toro, 2006), the equivalent combined model, and the random-regression approximation of Strandén and Mäntysaari (2013)."),
    (name = :ex_14_3, chapter = 14, file = "chapter-14/ex-3.jl",
     title = "Examples 14.3 and 14.4 – purebred and crossbred performance with breed of origin of alleles (Christensen et al., 2014): pedigree-based breed-specific partial relationship matrices, then their genomic counterparts from phased crossbred genotypes."),
    (name = :ex_15_1, chapter = 15, file = "chapter-15/ex-1.jl",
     title = "Example 15.1 – threshold (probit) sire model for calving ease in three ordered categories (Gianola and Foulley, 1983), with category probabilities by sire and a linear-model comparison."),
    (name = :ex_15_2, chapter = 15, file = "chapter-15/ex-2.jl",
     title = "Example 15.2 – joint analysis of birth weight (linear) and calving difficulty (binary threshold) (Foulley et al., 1983), and the bivariate linear model."),
    (name = :ex_17_1, chapter = 17, file = "chapter-17/ex-1.jl",
     title = "Sections 17.3 and 17.5 – sire variance by Henderson's method 3 and by fitting the canonical sums of squares (OLS and iterated GLS)."),
    (name = :ex_17_2, chapter = 17, file = "chapter-17/ex-2.jl",
     title = "Section 17.7 – REML for the animal model by average information (Table 17.3) and by the EM-type updates of Eqns 17.5–17.6."),
    (name = :ex_18_1, chapter = 18, file = "chapter-18/ex-1.jl",
     title = "Example 18.1 – Gibbs sampling for the univariate animal model: the first round replayed with the book's normal deviates, then full chains."),
    (name = :ex_18_2, chapter = 18, file = "chapter-18/ex-2.jl",
     title = "Example 18.2 – block Gibbs sampling for a multivariate animal model (WWG and PWG of Example 6.1), with inverse-Wishart updates of R and G."),
    (name = :ex_19_1, chapter = 19, file = "chapter-19/ex-1.jl",
     title = "Examples 19.1 and 19.2 – Jacobi (relaxation 0.8 on animal equations) and Gauss–Seidel iteration on the MME of Example 4.1."),
    (name = :ex_19_3, chapter = 19, file = "chapter-19/ex-3.jl",
     title = "Examples 19.3, 19.4 and 19.6 – iteration on data for the animal model: Gauss–Seidel without and with unknown parent groups, and the preconditioned conjugate gradient with a matrix-free C·d."),
    (name = :ex_19_5, chapter = 19, file = "chapter-19/ex-5.jl",
     title = "Example 19.5 – reduced animal model with maternal effects (Example 8.2) solved by Gauss–Seidel iteration."),
]

export EXAMPLES, list_examples, run_examples, check
export ex_04_1, ex_04_2, ex_04_3, ex_04_4, ex_05_1, ex_05_2, ex_06_1, ex_06_2, ex_06_3, ex_06_4, ex_06_5, ex_08_1, ex_08_2, ex_09_1, ex_10_1, ex_10_2, ex_10_3, ex_11_1, ex_11_2, ex_11_3, ex_11_6, ex_11_7, ex_11_8, ex_11_9, ex_11_13, ex_12_1, ex_12_2, ex_13_1, ex_13_2, ex_13_3, ex_13_5, ex_14_1, ex_14_3, ex_15_1, ex_15_2, ex_17_1, ex_17_2, ex_18_1, ex_18_2, ex_19_1, ex_19_3, ex_19_5

end # module Mrode4edExamples
