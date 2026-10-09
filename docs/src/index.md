# Mrode4edExamples.jl

*Julia solutions and verified numerical reproductions for worked examples in quantitative genetics and animal breeding.*

`Mrode4edExamples.jl` provides comprehensive, reproducible Julia implementations of the worked examples from:

> **Mrode, R. A., & Pocrnic, T. (2023).**  
> *Linear Models for the Prediction of the Genetic Merit of Animals* (4th ed.).  
> CABI Publishing, Wallingford, UK.

The package is built upon the core **[JuliaBnG](https://github.com/JuliaBnG)** ecosystem:
- **[Breeding.jl](https://github.com/JuliaBnG/Breeding.jl)**: General sparse mixed model equations (MME) solver, genomic prediction engines (BayesA/B/C/Cπ), threshold and non-linear generalized linear models, variance component estimation (Henderson Method 3, EM-REML, AI-REML), Gibbs samplers, and iteration on data (IOD / PCG).
- **[RelationshipMatrices.jl](https://github.com/JuliaBnG/RelationshipMatrices.jl)**: Fast pedigree-based relationship inverses ($A^{-1}$) with unknown parent groups (QP) and sire–MGS pedigrees, genomic relationship matrices ($G$), single-step inverse ($H^{-1}$), APY ($G^{-1}$), dominance relationship matrices ($D$), epistasis, and breed-of-origin matrices.

---

## Key Features

- **42 Worked Examples Across 14 Chapters**: Complete coverage of univariate, multivariate, maternal, social interaction, random regression, genomic, crossbreeding, threshold, variance-component, Bayesian, and iterative methods.
- **303 Exact Numerical Verifications**: Every example includes automated checks verifying computed estimates against book tables and textbook formulas.
- **Documented Corrections & Typo Analysis**: Transparently catalogues and resolves typographical slips, rounding discrepancies, and sign inconsistencies present in the 4th edition text.
- **Instructional & Research-Ready**: Serves as both a reference standard for students and a benchmark testbed for statistical geneticists.

---

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaBnG/Mrode-4ed-examples.jl")
```

The package depends on `Breeding.jl` and `RelationshipMatrices.jl`.

---

## Quick Example

```julia
using Mrode4edExamples

# Run a specific example (e.g. Chapter 4, Example 4.1: Univariate Animal Model)
res = ex_04_1()

# Inspect fixed effects and EBVs
res.b
res.a
```

To list all available examples or run batch verifications:

```julia
# View catalog of all examples
list_examples()

# Run all examples in a chapter
run_examples(11)

# Run entire test suite (all 42 examples and 303 checks)
run_examples(; quiet = true)
```

---

## Monograph & Ecosystem Navigation

Part of the **JuliaBnG Monograph Series**:
- **[BnGStructs.jl](https://juliabng.github.io/BnGStructs/)**: Foundational genomic data structures and bit-matrices.
- **[FisherWright.jl](https://juliabng.github.io/FisherWright/)**: Forward-time population genetic simulations.
- **[RelationshipMatrices.jl](https://juliabng.github.io/RelationshipMatrices/)**: Kinship and genomic relationship computing.
- **[Breeding.jl](https://juliabng.github.io/Breeding/)**: Mixed linear models and genomic selection.
- **[BnGsim.jl](https://juliabng.github.io/BnGsim/)**: Ecosystem orchestration and pipelines.
- **[Mrode4edExamples.jl](https://juliabng.github.io/Mrode-4ed-examples/)**: Applied linear models and benchmark reproductions.
