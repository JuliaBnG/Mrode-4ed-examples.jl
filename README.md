# Mrode4edExamples.jl

Julia solutions to the worked examples of Mrode and Pocrnic (2023),
*Linear Models for the Prediction of the Genetic Merit of Animals* (4th
edition), written for students and instructors and built on the
JuliaBnG packages:

- **[RelationshipMatrices.jl](https://github.com/JuliaBnG/RelationshipMatrices.jl)**
  – pedigree $A$ and $A^{-1}$ (including unknown parent groups and
  sire–MGS pedigrees), group contributions $Q$, genomic $G$,
  single-step $H^{-1}$, APY $G^{-1}$, tuning of $G$ to $A_{22}$,
  pedigree and genomic dominance $D$, epistatic $G_{AA}$, and
  breed-specific partial relationship matrices;
- **[Breeding.jl](https://github.com/JuliaBnG/Breeding.jl)** – a general
  sparse MME engine, EBV partitioning and reliabilities, genomic
  prediction, Bayes A/B/C/C$\pi$, threshold models, Henderson's Method
  3, EM- and AI-REML, Gibbs samplers, covariance functions, and
  iterative solvers (including iteration on data and matrix-free PCG).

[![Documentation](https://img.shields.io/badge/docs-juliabng.github.io-blue.svg)](https://juliabng.github.io/#applications)

The full documentation and API reference are available at
[juliabng.github.io/Mrode-4ed-examples](https://juliabng.github.io/Mrode-4ed-examples/).

## How the Examples Are Written

Each example is designed to be read alongside the textbook:

1. **Tabular data and results (`DataFrames.jl`)**: Pedigrees and
   phenotypic records are kept in `DataFrame`s—usually one row per
   animal, with animal IDs equal to the row numbers `1:n` and `missing`
   for unrecorded founders. As the example progresses, estimated
   breeding values, reliabilities, partitions, and model comparisons are
   added directly as columns of that table so you can inspect the data
   and the solutions side by side.
2. **Explicit block-matrix equations**: Rather than treating the solver
   as a black box, every example that solves Mixed Model Equations (MME)
   first writes them out directly in Julia's block-matrix notation—using
   the textbook's variance-ratio ($\lambda$) form when $R =
   I\sigma_e^2$, the general $R^{-1}$ form when residuals are
   heterogeneous, Kronecker products for multi-trait models, or $T'T +
   \Sigma$ when several random effects share records—solves `lhs \ rhs`,
   and verifies that the explicit system matches `mme(...)`.
3. **Step-by-step derivations checked against the library**: Key
   algorithms (RAM back-solving, the QP-transformed $A^{*-1}$,
   Cockerham's dominance rule, GBLUP back-solving and selection index,
   $G$-matrix tuning and $H^{-1}$, the APY inverse from its blocks, WLS
   covariance functions, Method 3, REML updates, single-site Gibbs,
   Jacobi, Gauss–Seidel, and matrix-free PCG) are written out in a few
   lines of Julia and checked against the corresponding library
   functions.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaBnG/Mrode-4ed-examples.jl")
```

Its dependencies, including `Breeding` (≥ 0.1.6) and
`RelationshipMatrices`, are installed from the General registry.

## Usage

```julia
using Mrode4edExamples

list_examples()                     # print the catalogue
ex_04_1()                           # run one example: prints, checks, returns results
run_examples(11)                    # run all examples of chapter 11
run_examples([:ex_12_1, :ex_12_2])  # run selected examples
run_examples(; quiet = true)        # run everything, summary only
```

Each example prints its intermediate steps and tables, compares its
results with the values printed in the book (`ok` / `DIFFERS`), and
returns a `NamedTuple` (or `DataFrame` / result struct) for interactive
exploration. `Pkg.test("Mrode4edExamples")` runs every example as a
test: 42 examples with 392 numerical checks, all passing.

The source files live in `src/chapter-CC/ex-N.jl`, with one function
`ex_CC_N` per file; `EXAMPLES` lists all of them with their titles.

## Examples

| Folder (`src/`) | Script | Book Sections & Topics |
|---|---|---|
| chapter-04 | ex-1 … ex-4 | 4.1–4.3 animal model, accuracies, PA/YD/PC, DYD; 4.4 sire model; 4.5 RAM; 4.6 unknown parent groups (QP) |
| chapter-05 | ex-1, ex-2 | 5.1–5.2 repeatability model and DYD; 5.3 common environment (litter effects) |
| chapter-06 | ex-1 … ex-5 | 6.1–6.4 multivariate models (equal design, missing records, unequal design, reliabilities, partition, DYD); 6.5.1 zero residual covariance; 6.5.2 MACE |
| chapter-08 | ex-1, ex-2 | 8.1–8.2 maternal animal model; 8.3 maternal RAM and back-solving |
| chapter-09 | ex-1 | 9.2–9.4 social interaction (associative effects, random group effect, correlated residuals, partition) |
| chapter-10 | ex-1 … ex-3 | 10.2 fixed regression; 10.3 random regression, 305-d values, reliabilities, partition; 10.4 & 10.3.5 covariance functions, linear splines |
| chapter-11 | ex-1 … ex-13 | 11.3 fixed SNPs; 11.4.1 SNP-BLUP; 11.4.2–11.4.5 & 11.9 GBLUP, back-solving, selection index, reliabilities; 11.4.6 base allele frequencies; 11.5 polygenic effects; 11.6 haplotypes; 11.7 Bayes A/B/C/C$\pi$; 11.8 multivariate GBLUP |
| chapter-12 | ex-1, ex-2 | 12.1 ssGBLUP ($H^{-1}$, blending and tuning); 12.4 APY $G^{-1}$ |
| chapter-13 | ex-1 … ex-5 | 13.2–13.3 pedigree dominance and total genetic merit; 13.4 genomic dominance and inbreeding depression; 13.5 epistasis |
| chapter-14 | ex-1, ex-3 | 14.2–14.4 partial relationship matrices, combined model, RR approximation; 14.7 breed of origin of alleles (pedigree and genomic) |
| chapter-15 | ex-1, ex-2 | 15.2 threshold (probit) sire model; 15.3 joint binary–quantitative analysis |
| chapter-17 | ex-1, ex-2 | 17.3 & 17.5 Henderson's Method 3 and canonical sums of squares; 17.7 AI- and EM-REML |
| chapter-18 | ex-1, ex-2 | 18.2 univariate Gibbs (book's first round replayed, full chains); 18.3 multivariate block Gibbs with data augmentation |
| chapter-19 | ex-1 … ex-5 | 19.4 Jacobi and Gauss–Seidel on the MME; 19.5.1–19.5.2 & 19.6 iteration on data, groups, and matrix-free PCG; 19.5.3 maternal RAM by Gauss–Seidel |

Chapters 2, 3, 7, and 16 have no folder (as in the R and BLUPF90
collections); their examples are not covered.

## Discrepancies with the Book

Where a printed number in the textbook has a typographical or rounding
slip, the script explains why and verifies the corrected value (see
[Textbook Discrepancies](docs/src/manual/discrepancies.md) for full
details):

- **6.3** The DYD of sire 1 uses $\mathrm{YD}_{4,1} = 25.7$, whereas
  $201 - 175.7 = 25.3$.
- **6.4** Animal 6 is printed as $(-5.012, -2.098)$, a copy of animal 5
  with the wrong signs; dam 6 has a high-performing son (animal 11), so
  her evaluations are $(+5.012, +2.098)$.
- **6.5** $W_1$ of bull 3 has a spurious $(1,2)$ entry; Eqn 6.21 needs a
  `+` sign; the progeny contribution of bull 2 omits the maternal
  granddam group (G5) of his grandson.
- **10.1** The printed lactation curve from DIM 174 onward does not
  match $\Phi\hat{b}_2$.
- **10.2** $k'Pk$ is printed as $323463$, whereas the book's own $t$ and
  $P$ give $320126$; $t$ sums 4-decimal rounded Legendre coefficients.
- **10.3** In Section 10.3.5, the linear spline table places the larger
  weight on the farther knot rather than standard linear interpolation.
- **11.1** The selection candidates 21–26 are labelled 25–30 in the
  table of results.
- **11.6** The GLS base mean $0.575$ is not reproduced ($0.526$ with
  $(A^{-1})_{22}$ or $0.590$ with $A_{22}^{-1}$).
- **11.7** The columns headed "DGV" in Table 11.4 actually tabulate
  $\mathrm{DGV} + \hat{u}$, and $\hat{u}$ uses the five-generation $A$
  printed in Example 11.3.
- **11.8** Animal 5's block-2 pseudo-SNPs are printed as `01000` instead
  of `01001`; the printed $G_h$, $2\sum pq = 4.42$, and haplotype GBLUP
  follow from that typo, and the correlation between the book's own SNP
  and haplotype solutions is $0.992$, not $0.999$.
- **11.9** The first starting residual $\hat{e}_1$ is $-1.338$, not
  $-1.388$.
- **11.10** The BayesB posterior SNP variances ($\approx 0.30$ in the
  book) are not reproduced (here $\approx 0.12$ on average, stable over
  seeds and chain lengths); all other posterior means and the
  BayesC$\pi$ posterior $\pi = 0.51$ agree.
- **12.2** $G_{\mathrm{APY}}^{-1}$ is printed in APY order (core bulls
  first); bull 17's DGV omits its minus sign.
- **14.2** Table 14.3 lists the weighted contributions $\sqrt{f_p} u_p$
  rather than $u_p$.
- **14.3** Solutions agree within $0.005$, though the book's values are
  slightly asymmetric for exchangeable founders; **14.4** the printed
  $G^{(1)}$ does not follow from Eqn 14.25, whereas Table 14.9 does.
- **15.1** The linear-model column cannot be reproduced from Table 15.2;
  sire 4's category probabilities are off by $0.003$.
- **17.5** The unweighted OLS fit ($0.143, 0.079$) and "estimated
  variances" ($0.216, 0.234$) are not reproduced (OLS gives
  $0.151, 0.062$); the iterated GLS estimates are. **17.7** $\hat{a}_4$
  at the starting values is $+0.254$, not $-0.254$.
- **18.1** With flat/improper priors ($\nu = -2, S^2 = 0$) and only five
  records, the joint posterior is improper near $\sigma_a^2 = 0$; proper
  scaled inverse-$\chi^2$ priors are used for the full chain.
- **19.3** The column headed "At convergence" is round 20, whereas the
  $10^{-7}$ criterion is met at round 9. **19.4** The constrained column
  is not the converged solution of Example 4.4. **19.5** The second
  fixed factor is called "Pen" in Table 8.1 and "Sex" in Example 19.5.
