# Catalogue of Chapters

This catalogue summarizes the quantitative genetic methodology and computational features implemented across each chapter of Mrode & Pocrnic (2023).

---

## Chapter 4: Best Linear Unbiased Prediction of Breeding Values

Covers basic linear mixed model equations (MME) for single-trait evaluations:
- **`ex_04_1` (Sections 4.1–4.3)**: Univariate animal model for
  pre-weaning gain (WWG), with pedigree, records and all solutions in
  one `DataFrame` (`ainv(ped)` uses only its `sire` and `dam` columns).
  The MME are written out explicitly in the general form, with
  $R^{-1}$ (heterogeneous residuals allowed), and in the book's form
  $[X'X\ X'Z;\ Z'X\ Z'Z + \lambda A^{-1}]$ for $R = I\sigma^2_e$; both
  are checked against `mme`. Also covers accuracy (prediction error
  variance $\mathrm{PEV}$, reliability $r^2$, $\mathrm{SEP}$) and the
  decomposition of breeding values into Parent Average
  ($\mathrm{PA}$), Yield Deviation ($\mathrm{YD}$), and Progeny
  Contribution ($\mathrm{PC}$).
- **`ex_04_2` (Section 4.4)**: Sire model on the data table of
  `ex_04_1`, with the sire pedigree derived from it and recoded for
  $A_s^{-1}$. The MME $[X'X\ X'Z_s;\ Z_s'X\ Z_s'Z_s + \alpha_s A_s^{-1}]$
  are written out and checked against `mme`. Sire solutions are shown
  next to $\frac{1}{2}\hat{a}$ of the animal model: $s = \frac{1}{2}a$,
  but here the estimates even rank the sires differently.
- **`ex_04_3` (Section 4.5)**: Reduced Animal Model (RAM) on the data
  table of `ex_04_1`. Records of non-parents carry their Mendelian
  sampling in the residual, so $R \ne I\sigma^2_e$ and the MME are
  written out in the general form
  $[X'R^{-1}X\ X'R^{-1}W;\ W'R^{-1}X\ W'R^{-1}W + A_p^{-1}/\sigma^2_a]$.
  Non-parents are back-solved explicitly (Eqn 4.26), and the solutions
  equal those of the full animal model with 8 instead of 10 equations.
- **`ex_04_4` (Section 4.6)**: Animal model with unknown parent groups
  (UPG) on the data table of `ex_04_1`, with unknown parents replaced by
  negative group codes. $A^{*-1}$ is built from its blocks
  $[A^{-1}\ -A^{-1}Q;\ -Q'A^{-1}\ Q'A^{-1}Q]$ and checked against
  `ainv_upg`; the QP-transformed MME are written out and checked
  against `mme`, and $\hat{a}^* = \hat{a} + Q\hat{g}$ is confirmed with
  the explicit group model.

---

## Chapter 5: Models for Repeated Records

Extends the mixed model to traits recorded multiple times across an animal's lifetime:
- **`ex_05_1` (Section 5.1)**: Univariate repeatability animal model
  with permanent environmental effects ($pe$). Uses two tables, the
  pedigree (one row per animal) and the records (one row per
  lactation); yield deviations come from `groupby`/`combine` and are
  joined onto the pedigree. The MME with $\lambda_a A^{-1}$ and
  $\lambda_{pe} I$ are written out and checked against `mme`, and the
  rank deficiency of $X$ explains the two constraints. Computes
  daughter yield deviations ($\mathrm{DYD}$) and the partition of EBV.
- **`ex_05_2` (Section 5.3)**: Common environmental effect ($c^2$)
  shared by full-sibs or littermates, on one data table with a litter
  column. The MME with $\lambda_a A^{-1}$ and $\lambda_c I$ are written
  out and checked against `mme`; litter solutions are kept in a table
  of litters and joined back onto the piglets.

---

## Chapter 6: Multivariate Models

Simultaneous genetic evaluation of multiple genetically correlated traits:
- **`ex_06_1` (Section 6.1)**: Bivariate animal model for pre-weaning
  gain (WWG) and post-weaning gain (PWG) with complete records, on one
  data table. The MME are written out with `kron`,
  $[R_0^{-1}\otimes X'X\ R_0^{-1}\otimes X'Z;\ R_0^{-1}\otimes Z'X\
  R_0^{-1}\otimes Z'Z + G_0^{-1}\otimes A^{-1}]$, and checked against
  the general set-up with `mt_stack`, `mt_blocks` and `mt_rinv`.
  Multi-trait and univariate EBV, PEV and reliabilities are added as
  columns.
- **`ex_06_2` (Section 6.2)**: Unequal design and missing traits across animals.
- **`ex_06_3` (Section 6.3)**: Partition of multivariate EBVs and multi-trait reliability calculations.
- **`ex_06_4` (Section 6.4)**: Multivariate analysis with zero residual covariance (e.g. traits measured on different animals).
- **`ex_06_5` (Section 6.5)**: Multiple Across Country Evaluation (MACE) formulation for international dairy evaluations.

---

## Chapter 8: Maternal Effects Models

Models accounting for both the direct genetic ability of an animal and the maternal genetic/environmental contribution of its dam:
- **`ex_08_1` (Section 8.1)**: Maternal animal model with correlated direct ($a$) and maternal ($m$) genetic effects ($\sigma_{am}$ covariance).
- **`ex_08_2` (Section 8.2)**: Maternal Reduced Animal Model (RAM).

---

## Chapter 9: Social Interaction Models

Evaluates indirect genetic effects (IGE) / associative effects in group-housed animals:
- **`ex_09_1` (Section 9.1)**: Mixed model with direct breeding values ($a_D$) and social breeding values ($a_S$) under pen-level housing and correlated group environmental effects. Partition of total breeding values ($TBV_i = a_{D,i} + (n-1)a_{S,i}$).

---

## Chapter 10: Random Regression Models

Models continuous longitudinal or trajectory traits (e.g. test-day milk yields):
- **`ex_10_1` (Section 10.1)**: Fixed regression test-day model with Ali-Schaeffer lactation curves.
- **`ex_10_2` (Section 10.2)**: Random regression test-day model using orthogonal Legendre polynomials for animal genetic and permanent environmental trajectories. Calculation of 305-day cumulative lactation breeding values and reliabilities.
- **`ex_10_3` (Section 10.4)**: Random regression using linear splines and covariance functions.

---

## Chapter 11: Genomic Selection & The Bayesian Alphabet

Comprehensive genomic evaluation incorporating high-density marker panels:
- **`ex_11_1` (Section 11.2)**: Fixed regression on individual QTL/SNP genotypes.
- **`ex_11_2` (Section 11.3)**: Ridge regression BLUP / SNP-BLUP estimating individual SNP allele substitution effects.
- **`ex_11_3` (Sections 11.4–11.5)**: Genomic BLUP (GBLUP) using VanRaden's first genomic relationship matrix ($G$), back-solving marker effects from GEBVs, selection index, and genomic reliabilities.
- **`ex_11_6` (Section 11.6)**: Estimation of base population allele frequencies via Generalized Least Squares (GLS).
- **`ex_11_7` (Section 11.7)**: GBLUP with residual polygenic effect blending ($G^* = w G + (1-w) A_{22}$).
- **`ex_11_8` (Section 11.8)**: Haplotype-based genomic prediction models.
- **`ex_11_9` (Sections 11.9–11.12)**: The Bayesian Alphabet:
  - **BayesA**: Student-$t$ prior via locus-specific scaled inverse chi-square variance draws.
  - **BayesB**: Point mass mixture prior (variable selection with locus-specific variances).
  - **BayesC**: Point mass mixture prior with a common genetic variance.
  - **BayesC$\pi$**: Joint sampling of marker inclusion probability $\pi$.
- **`ex_11_13` (Section 11.13)**: Multivariate GBLUP for multi-trait genomic predictions.

---

## Chapter 12: Single-Step GBLUP and APY

Integration of genotyped and non-genotyped animals into unified national evaluations:
- **`ex_12_1` (Section 12.1)**: Single-step GBLUP (ssGBLUP) with the Aguilar/Legarra $H^{-1}$ matrix:
  ```math
  H^{-1} = A^{-1} + \begin{bmatrix} 0 & 0 \\ 0 & G^{-1} - A_{22}^{-1} \end{bmatrix}
  ```
- **`ex_12_2` (Section 12.2)**: Algorithm for Proven and Young (APY) sparse genomic inversion, partitioning genotyped animals into core and non-core subsets.

---

## Chapter 13: Non-Additive Genetic Models

Partitioning non-additive genetic variation for commercial crossing and mating allocation:
- **`ex_13_1`–`ex_13_2` (Sections 13.1–13.2)**: Pedigree-based dominance relationship matrix ($D$) and dominance BLUP.
- **`ex_13_3` (Sections 13.3–13.4)**: Genomic dominance relationship matrices and estimation of genomic inbreeding depression.
- **`ex_13_5` (Section 13.5)**: Additive-by-additive ($A \odot A$ or $G \odot G$) epistatic relationship models.

---

## Chapter 14: Crossbreeding & Breed of Origin of Alleles

Linear models for multibreed populations and commercial crossbreds:
- **`ex_14_1` (Sections 14.1–14.2)**: Partial relationship matrices for multi-breed evaluations and reduced-rank approximations.
- **`ex_14_3` (Sections 14.3–14.4)**: Breed of Origin of Alleles (BOA) evaluations using pedigree and genomic tracking of breed-specific alleles.

---

## Chapter 15: Non-Linear & Threshold Models

Evaluation of categorical and disease resistance traits:
- **`ex_15_1` (Section 15.1)**: Gianola & Foulley threshold animal model for binary/ordinal traits on the underlying liability scale.
- **`ex_15_2` (Section 15.2)**: Joint analysis of continuous and binary threshold traits.

---

## Chapter 17: Variance Component Estimation

Estimation of genetic and residual variance components:
- **`ex_17_1` (Sections 17.3–17.5)**: Henderson's Method 3 and canonical reduction of sums of squares.
- **`ex_17_2` (Section 17.7)**: Restricted Maximum Likelihood (REML) via Expectation-Maximization (EM-REML) and Average Information (AI-REML).

---

## Chapter 18: Bayesian Analysis & Gibbs Sampling

Markov Chain Monte Carlo (MCMC) Bayesian parameter estimation:
- **`ex_18_1` (Section 18.1)**: Univariate animal model Gibbs sampler with replayed pseudo-random deviates and posterior chain convergence.
- **`ex_18_2` (Section 18.2)**: Multivariate block Gibbs sampler with inverse-Wishart covariance updates for $R$ and $G$.

---

## Chapter 19: Iterative Solvers for Large-Scale Systems

High-performance linear solvers for systems with millions of equations:
- **`ex_19_1` (Sections 19.1–19.2)**: Stationary iterative solvers (Jacobi with relaxation and Gauss-Seidel).
- **`ex_19_3` (Sections 19.3, 19.4, 19.6)**: Iteration on Data (IOD) for massive pedigrees:
  - Gauss-Seidel on data without and with genetic groups.
  - Preconditioned Conjugate Gradient (PCG) on data using matrix-free operations ($C \cdot d$).
- **`ex_19_5` (Section 19.5)**: Reduced Animal Model with maternal effects solved via Gauss-Seidel iteration.
