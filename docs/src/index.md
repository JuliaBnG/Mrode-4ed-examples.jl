# Mrode4edExamples.jl

*Worked examples in animal breeding and quantitative genetics, written
step by step in Julia.*

`Mrode4edExamples.jl` is a companion package to the textbook:

> **Mrode, R. A., & Pocrnic, T. (2023).**  
> *Linear Models for the Prediction of the Genetic Merit of Animals*
> (4th ed.).  
> CABI Publishing, Wallingford, UK.

It is written for students learning linear mixed models and genomic
prediction, and for instructors teaching from Mrode & Pocrnic (2023).
The worked examples of Chapters 4–6, 8–15, and 17–19 are self-contained
Julia functions (one function may cover several closely related
examples). Each sets up the data in a `DataFrame`, writes out the Mixed
Model Equations (MME) or the key algorithm and intermediate matrices in
readable Julia notation, checks them against the general routines in
the **[JuliaBnG](https://github.com/JuliaBnG)** packages, and verifies
the numerical results against the tables printed in the book:

- **[RelationshipMatrices.jl](https://github.com/JuliaBnG/RelationshipMatrices.jl)**:
  Pedigree numerator relationship matrix $A$ and its inverse $A^{-1}$
  (including unknown parent groups and sire–MGS pedigrees), genetic
  group contributions $Q$, genomic relationship matrix $G$, single-step
  inverse $H^{-1}$, APY inverse $G_{\mathrm{APY}}^{-1}$, tuning of $G$
  to $A_{22}$, pedigree and genomic dominance $D$, epistatic $G_{AA}$,
  and breed-specific partial relationship matrices.
- **[Breeding.jl](https://github.com/JuliaBnG/Breeding.jl)**: General
  sparse MME assembly and solvers, prediction error variances,
  reliabilities and EBV partitions, genomic prediction (SNP-BLUP,
  GBLUP, BayesA/B/C/C$\pi$), threshold and joint binary–quantitative
  models, variance component estimation (Henderson's Method 3, EM-REML,
  AI-REML), Gibbs samplers, covariance functions, and iterative solvers
  (Jacobi, Gauss–Seidel, iteration on data, and matrix-free PCG).

---

## Highlights

- **42 Worked Examples Across 14 Chapters**: Covers animal, sire, and
  reduced animal models, unknown parent groups, repeatability and
  litter models, multivariate models and MACE, maternal and social
  interaction models, random regression and covariance functions,
  genomic selection and the Bayesian alphabet, single-step GBLUP and
  APY, dominance and epistasis, crossbreeding and breed of origin of
  alleles, threshold models, REML, Gibbs sampling, and iteration on
  data.
- **Written for Students**: Rather than hiding the algebra behind a
  black-box solver, each script that solves MME first writes them out
  explicitly as a Julia block matrix (`[X'X X'Z; Z'X Z'Z + λ * Ai]`,
  `[X'Ri*X X'Ri*W; ...]`, Kronecker forms, or $T'T + \Sigma$) and
  compares it with `mme(...)`. Pedigrees, phenotypes, and solutions are
  stored together in `DataFrame`s so you can inspect data and
  predictions side by side.
- **392 Automated Numerical Checks**: The examples verify both the
  equivalence of the hand-written equations or algorithms with the
  library routines and the agreement of the solutions with the numbers
  printed in the textbook.
- **Documented Textbook Errata**: Where a printed table in the 4th
  edition contains a typographical or arithmetic slip, the example
  explains the discrepancy and checks the corrected value (see
  [Textbook Discrepancies](manual/discrepancies.md)).

---

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaBnG/Mrode-4ed-examples.jl")
```

---

## Quick Start

```julia
using Mrode4edExamples

# Run Chapter 4, Example 4.1 (Univariate Animal Model)
res = ex_04_1()

# Inspect the pedigree table with records, EBVs, reliabilities, and PA/YD/PC
res.ped

# Inspect the fixed sex effects and breeding value vector
res.b
res.a
```

To browse the catalogue or run chapters in batch:

```julia
# List all 42 examples with their chapter and title
list_examples()

# Run all examples in Chapter 11
run_examples(11)

# Run the entire suite (42 examples, 392 checks) and print a summary
run_examples(; quiet = true)
```

---

## JuliaBnG Ecosystem

`Mrode4edExamples.jl` is part of the **JuliaBnG** series:
- **[BnGStructs.jl](https://juliabng.github.io/BnGStructs/)**: Genomic
  data structures and bit-packed genotype arrays.
- **[FisherWright.jl](https://juliabng.github.io/FisherWright/)**:
  Forward-time population genetic simulations.
- **[RelationshipMatrices.jl](https://juliabng.github.io/RelationshipMatrices/)**:
  Pedigree and genomic relationship matrices and inverses.
- **[Breeding.jl](https://juliabng.github.io/Breeding/)**: Linear mixed
  models and genomic evaluation.
- **[BnGsim.jl](https://juliabng.github.io/BnGsim/)**: Breeding program
  simulation pipelines.
- **[Mrode4edExamples.jl](https://juliabng.github.io/Mrode-4ed-examples/)**:
  Worked textbook examples and numerical benchmarks.
