# Mrode4edExamples.jl

Julia solutions to the worked examples of Mrode and Pocrnic (2023),
*Linear Models for the Prediction of the Genetic Merit of Animals*, 4th
edition, as a package built on the JuliaBnG packages:

- **[RelationshipMatrices.jl](https://github.com/JuliaBnG/RelationshipMatrices.jl)**
  – A, A⁻¹ (also with unknown parent groups and for sire–MGS
  pedigrees), group contributions Q, G, H⁻¹, APY G⁻¹, tuning of G to
  A₂₂, pedigree dominance D, epistatic and partial (breed-specific)
  relationship matrices;
- **[Breeding.jl](https://github.com/JuliaBnG/Breeding.jl)** – a general
  sparse MME engine, iterative solvers and iteration on data, genomic
  prediction, Bayes A/B/C/Cπ, threshold models, REML, Gibbs samplers
  and covariance functions.

[![Documentation](https://img.shields.io/badge/docs-juliabng.github.io-blue.svg)](https://juliabng.github.io/#applications)

The documentation and API reference are available at
[juliabng.github.io/Mrode-4ed-examples](https://juliabng.github.io/Mrode-4ed-examples/).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaBnG/Mrode-4ed-examples.jl")
```

The package depends on Breeding from its GitHub repository (see
`[sources]` in `Project.toml`) until its registration in General is
merged.

## Usage

```julia
using Mrode4edExamples

list_examples()                     # the catalogue
ex_04_1()                           # one example: prints, checks, returns results
run_examples(11)                    # all examples of chapter 11
run_examples([:ex_12_1, :ex_12_2])  # selected examples
run_examples(; quiet = true)        # everything, summary only
```

Each example prints its results and compares them with the values
printed in the book (`ok` / `DIFFERS`); `run_examples` summarises the
comparisons. `Pkg.test("Mrode4edExamples")` runs every example as a
test: currently 42 examples with 303 comparisons, all passing.

The sources are in `src/chapter-CC/ex-N.jl`, one function `ex_CC_N`
per file; `EXAMPLES` lists them with their titles.

## Examples

| Folder (`src/`) | Script | Book |
|---|---|---|
| chapter-04 | ex-1 … ex-4 | 4.1 animal model, accuracies, PA/YD/PC, PYD; 4.2 sire model; 4.3 RAM; 4.4 groups (QP) |
| chapter-05 | ex-1, ex-2 | 5.1 repeatability and DYD; 5.2 common environment |
| chapter-06 | ex-1 … ex-5 | 6.1–6.3 multivariate (missing records, unequal design, partition, reliabilities, DYD); 6.4 no residual covariance; 6.5 MACE |
| chapter-08 | ex-1, ex-2 | 8.1 maternal animal model; 8.2 RAM with maternal effects |
| chapter-09 | ex-1 | 9.1 social interaction (group effect and correlated residuals), partition |
| chapter-10 | ex-1 … ex-3 | 10.1 fixed regression; 10.2 random regression, 305-d values, reliabilities, partition; 10.4 covariance functions, linear splines |
| chapter-11 | ex-1 … ex-13 | 11.1 fixed SNPs; 11.2 SNP-BLUP; 11.3–11.5 GBLUP, back-solving, selection index, reliabilities; 11.6 base allele frequencies; 11.7 polygenic effects; 11.8 haplotypes; 11.9–11.12 Bayes A/B/C/Cπ; 11.13 multivariate GBLUP |
| chapter-12 | ex-1, ex-2 | 12.1 ssGBLUP; 12.2 APY |
| chapter-13 | ex-1 … ex-5 | 13.1–13.2 pedigree dominance; 13.3–13.4 genomic dominance, inbreeding depression; 13.5 epistasis |
| chapter-14 | ex-1, ex-3 | 14.1–14.2 partial relationship matrices, RR approximation; 14.3–14.4 breed of origin (pedigree, genomic) |
| chapter-15 | ex-1, ex-2 | 15.1 threshold model; 15.2 joint binary–quantitative analysis |
| chapter-17 | ex-1, ex-2 | 17.3–17.5 method 3 and canonical sums of squares; 17.7 AI- and EM-REML |
| chapter-18 | ex-1, ex-2 | 18.1 univariate Gibbs (book's first round replayed); 18.2 multivariate Gibbs |
| chapter-19 | ex-1 … ex-5 | 19.1 Jacobi; 19.2 Gauss–Seidel; 19.3–19.4 iteration on data, groups; 19.5 RAM maternal; 19.6 PCG on data |

Chapters 2, 3, 7 and 16 have no folder (as in the R and BLUPF90
collections) and are not covered.

## Discrepancies with the book

Where a printed value could not be reproduced, the script explains why
and checks the corrected value. In brief:

- **6.3** DYD of sire 1 uses YD₄₁ = 25.7 although 201 − 175.7 = 25.3.
- **6.4** Animal 6 is printed as a copy of animal 5 (wrong signs).
- **6.5** W₁ of bull 3 has a spurious (1,2) element; Eqn 6.21 needs `+`;
  the PC of bull 2 omits the grandson's MGD group.
- **10.1** The lactation curve from DIM 174 on is not Φb̂₂.
- **10.2** k′Pk is printed as 323463 (the book's own t and P give 320126);
  t uses 4-decimal Legendre coefficients.
- **11.1–11.2** Selection candidates are labelled 25–30; bull 17's
  SNP-BLUP DGV lacks its sign (12.1).
- **11.6** The GLS base mean 0.575 is not reproduced (0.526 or 0.590).
- **11.7** The "DGV" columns of Table 11.4 are DGV + polygenic effect,
  and the polygenic effects use the printed five-generation A.
- **11.8** Animal 5's pseudo-SNPs (block 2) are mistyped, and Gₕ, 2Σpq =
  4.42 and the haplotype GBLUP follow from the typo; the stated
  correlation 0.999 is 0.992 from the book's own solutions.
- **11.9** ê₁ is −1.338, not −1.388.
- **11.10, 11.12** The BayesB SNP variances (≈0.30) and the BayesCπ
  posterior π (0.51) are not reproduced (here ≈0.12–0.25 and ≈0.67,
  stable over seeds); the other posterior means agree.
- **12.2** G⁻¹_APY is printed in APY order; bull 17's DGV lacks its sign.
- **14.2** Table 14.3 lists √fₚ uₚ, not uₚ.
- **14.3** Solutions agree within 0.005 but are not symmetric for
  exchangeable founders; **14.4** the printed G⁽¹⁾ does not follow from
  Eqn 14.25 although Table 14.9 does.
- **15.1** The linear-model column is not reproducible from Table 15.2;
  sire 4's probabilities are off by 0.003.
- **17.5** The unweighted fit (0.143, 0.079) and the "estimated
  variances" (0.216, 0.234) are not reproduced; the iterated GLS
  estimates are. **17.7** â₄ is +0.254.
- **18.1** With the uniform priors and five records the posterior is
  improper near σ²a = 0; proper priors are used for the chain.
- **19.3** "At convergence" is round 20; the 10⁻⁷ criterion is met at
  round 9. **19.4** The constrained column is not the converged
  solution of Example 4.4.
