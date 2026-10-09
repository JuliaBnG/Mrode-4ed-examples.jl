# Overview & Quick Start

## Quantitative Genetics with Linear Models

Predicting the genetic merit of animals and plants from phenotypic records, pedigrees, and high-density genomic marker data is fundamentally an exercise in linear mixed model methodology. The seminal textbook by R. A. Mrode and T. Pocrnic (2023), *Linear Models for the Prediction of the Genetic Merit of Animals* (4th ed.), provides a pedagogical and computational standard across animal breeding curricula worldwide.

`Mrode4edExamples.jl` provides standard, reproducible Julia implementations of the worked examples in the 4th edition. Each example is designed to:
1. Formulate the statistical model precisely using modern Julia and the `JuliaBnG` ecosystem (`Breeding.jl` and `RelationshipMatrices.jl`).
2. Construct the Mixed Model Equations (MME) in sparse format.
3. Solve for best linear unbiased estimates (BLUE) of fixed effects and best linear unbiased predictions (BLUP) of random genetic effects.
4. Compute prediction error variances (PEV), accuracies, reliabilities, and partitions of breeding values (parent average, yield deviation, progeny contribution).
5. Numerically verify calculated outputs against the textbook's published tables and note any errata.

---

## Architecture & Workflow

Every example in the book is encapsulated as a callable Julia function named `ex_CC_N()`, where `CC` is the two-digit chapter number and `N` is the sequential script or example number:

| Function | Chapter | Example Title |
|:---|:---:|:---|
| `ex_04_1()` | 4 | Univariate animal model, accuracy, and partition of EBV |
| `ex_04_2()` | 4 | Sire model |
| `ex_04_3()` | 4 | Reduced Animal Model (RAM) |
| `ex_04_4()` | 4 | Animal model with unknown parent groups (QP) |
| `ex_05_1()` | 5 | Repeatability animal model and daughter yield deviations (DYD) |
| `ex_06_1()` | 6 | Multivariate animal model with equal design |
| `ex_11_2()` | 11 | SNP-BLUP marker regression |
| `ex_11_3()` | 11 | Genomic BLUP (GBLUP) and equivalence to SNP-BLUP |
| `ex_12_1()` | 12 | Single-step GBLUP (ssGBLUP) |
| `ex_12_2()` | 12 | Algorithm for Proven and Young (APY) |
| `ex_13_1()` | 13 | Pedigree-based dominance model |
| `ex_14_3()` | 14 | Breed of origin of alleles (BOA) |
| `ex_15_1()` | 15 | Threshold model for categorical traits |
| `ex_17_2()` | 17 | Average Information (AI) and EM-REML |
| `ex_18_1()` | 18 | Gibbs sampling for univariate animal model |
| `ex_19_3()` | 19 | Iteration on data (IOD) for animal model |

Each function returns a `NamedTuple` containing:
- `b`: Fixed effect solutions (BLUE).
- `a`: Genetic effect solutions (BLUP or EBV/GEBV).
- Specific intermediate structures, such as the coefficient matrix inverse $C$, partitions, reliability vectors, or variance components.

---

## Verifications via `check`

The helper function `check(label, got, ref; atol = 1e-3)` compares calculated matrices or vectors against reference numbers printed in the book.

```julia
using Mrode4edExamples

# Run a single example: prints steps, comparisons, and returns data
res = ex_04_1()
```

Sample output:
```text
Example 4.1: animal model, pre-weaning gain of beef calves
  Fixed effects (sex)                              ok (max |Δ| = 1.05e-04)
  EBV (all 8 animals)                              ok (max |Δ| = 2.15e-04)
  Reliabilities (r²)                               ok (max |Δ| = 3.50e-04)
  Partition weights (w1, w2, w3)                   ok (max |Δ| = 4.12e-04)
```

If an output diverges from the textbook's printed number due to a known book error or rounding limitation, the comparison evaluates against the corrected theoretical value, and details are logged in [Textbook Discrepancies](discrepancies.md).

---

## Batch Testing & Catalogue Query

You can inspect all available examples using `list_examples()`:

```julia
list_examples()
```

To run a subset of examples or evaluate test suites:

```julia
# Run all examples belonging to Chapter 11
run_examples(11)

# Run a list of specific examples
run_examples([:ex_12_1, :ex_12_2])

# Run all 42 examples quietly, summarizing only passes and failures
results = run_examples(; quiet = true)
println("Total checks: ", results.checks, ", Failed: ", results.failed)
```
