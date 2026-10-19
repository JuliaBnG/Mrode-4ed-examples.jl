# Catalogue of Chapters

This guide walks through the worked examples chapter by chapter,
highlighting both the quantitative-genetic models in Mrode & Pocrnic
(2023) and the way each calculation is set up in Julia using
`DataFrame`s and explicit block matrices.

---

## Chapter 4: Best Linear Unbiased Prediction of Breeding Values

Chapter 4 introduces the foundational Mixed Model Equations (MME) for a
single trait measured on beef calves (pre-weaning gain, WWG), comparing
the full animal model with the sire model, the Reduced Animal Model
(RAM), and unknown parent groups (UPG):

- **[`ex_04_1`](@ref) (Sections 4.1–4.3)**: Univariate animal model for
  pre-weaning gain. The pedigree, sex, and records of all 8 animals are
  kept in a single `DataFrame` (`ainv(ped)` uses only its `sire` and
  `dam` columns, while unrecorded founders have `missing` records). We
  write out the MME in two equivalent ways: first in the general form
  with $R^{-1}$ (which also works when residual variances are
  heterogeneous and gives $\mathrm{PEV} = C^{-1}$ directly), and then in
  the textbook's homogeneous-residual form
  $[X'X\ X'Z;\ Z'X\ Z'Z + \lambda A^{-1}]$ obtained by multiplying
  through by $\sigma_e^2$ ($\lambda = \sigma_e^2/\sigma_a^2 = 2$). Both
  are checked against `mme`. We then compute prediction error variances
  ($\mathrm{PEV}_i = d_i\sigma_e^2$), reliabilities
  ($r_i^2 = 1 - d_i\lambda$), standard errors of prediction
  ($\mathrm{SEP}$), and decompose each animal's EBV into Parent Average
  ($\mathrm{PA}$), Yield Deviation ($\mathrm{YD}$), and Progeny
  Contribution ($\mathrm{PC}$), along with the Daughter Yield Deviation
  ($\mathrm{DYD}$) of sire 3. All animal-level results are added as
  columns of `ped`.
- **[`ex_04_2`](@ref) (Section 4.4)**: Sire model on the same data table
  as `ex_04_1`. Only sires with recorded progeny (sires 1, 3, and 4)
  enter the model; their pedigree is extracted from `ped` (sire 4 is a
  son of sire 1, while dams are treated as unknown) and recoded `1:3` to
  build $A_s^{-1}$. Because a sire transmits half of his breeding value,
  $\sigma_s^2 = \frac{1}{4}\sigma_a^2 = 5$, and the remaining
  $\frac{3}{4}\sigma_a^2$ moves into the residual ($\sigma_e^2 = 55$,
  $\alpha_s = 11$). We write out
  $[X'X\ X'Z_s;\ Z_s'X\ Z_s'Z_s + \alpha_s A_s^{-1}]$, check it against
  `mme`, and compare $\hat{s}$ side by side with $\frac{1}{2}\hat{a}$
  from the animal model: although $s = \frac{1}{2}a$ in theory, the
  estimates differ—and even rank the three sires differently—because the
  sire model ignores sire 4's own phenotype and the merit of the mates.
- **[`ex_04_3`](@ref) (Section 4.5)**: Reduced Animal Model (RAM) on the
  data table of `ex_04_1`. Only parents (animals 1–6) are given
  equations in the MME; for non-parents (calves 7 and 8), their breeding
  value is written as $\frac{1}{2}(a_s + a_d) + m_i$ and the Mendelian
  sampling term $m_i$ is absorbed into the residual. Because parent and
  non-parent records now have different residual variances ($\sigma_e^2$
  vs $\sigma_e^2 + m_i\sigma_a^2$), $R \ne I\sigma_e^2$ and there is no
  single scalar $\lambda$: we write the MME in the general $R^{-1}$ form
  $[X'R^{-1}X\ X'R^{-1}W;\ W'R^{-1}X\ W'R^{-1}W + A_p^{-1}/\sigma_a^2]$,
  solve the 8 equations for fixed effects and parents, and back-solve
  non-parents 7 and 8 explicitly (Eqn 4.26), recovering the exact
  solutions of the 10-equation animal model.
- **[`ex_04_4`](@ref) (Section 4.6)**: Animal model with unknown parent
  groups (UPG) on the data table of `ex_04_1`. Unknown sires and dams
  are replaced by negative group codes (`-1` for group 1, `-2` for group
  2). We compute the group contribution matrix $Q$, build the
  Quaas–Pollak (QP) augmented inverse
  $A^{*-1} = [A^{-1}\ -A^{-1}Q;\ -Q'A^{-1}\ Q'A^{-1}Q]$ from its blocks
  (and check it against `ainv_upg`), and write out the $3 \times 3$
  block MME for $[\hat{b};\ \hat{a}^*;\ \hat{g}]$. Finally, we fit the
  explicit group model $y = Xb + ZQg + Za + e$ to verify that the QP
  solutions satisfy $\hat{a}^* = \hat{a} + Q\hat{g}$, and contrast them
  with adding $Q\hat{g}$ to $\hat{a}$ from a model that ignored groups.

---

## Chapter 5: Models for Repeated Records

Chapter 5 extends the mixed model to traits measured several times on
the same animal (repeatability models) or on littermates sharing a
common environment:

- **[`ex_05_1`](@ref) (Sections 5.1–5.2)**: Repeatability animal model
  for fat yield across two lactations, fitting both additive genetic
  effects ($a$) and permanent environmental effects ($pe$). Because each
  cow has two records, we use two tables: `ped` (one row per animal) and
  `rec` (one row per lactation). We write out the $3 \times 3$ block MME
  with $\lambda_a A^{-1}$ and $\lambda_{pe} I$, check it against `mme`,
  and inspect $\mathrm{rank}(X) = 4$ (with 6 columns) to see why two
  fixed-effect levels must be constrained to zero (HYS 1 occurs only in
  parity 1 and HYS 3 only in parity 2). Per-cow yield deviations are
  computed from `rec` with `groupby`/`combine`, joined onto `ped`, and
  used to partition the EBVs and compute the Daughter Yield Deviation
  ($\mathrm{DYD}$) of sire 1.
- **[`ex_05_2`](@ref) (Section 5.3)**: Animal model with common
  environmental (litter) effects ($c$) for piglet weaning weight. The
  pedigree, sex, litter, and weaning weight of all 15 pigs live in one
  `DataFrame`. We write out the $3 \times 3$ block MME with
  $\lambda_a A^{-1}$ and $\lambda_c I$, check it against `mme`, store
  the estimated litter effects in a `litters` table, and join them back
  onto the piglets in `ped`.

---

## Chapter 6: Multivariate Models

Chapter 6 shows how to evaluate genetically and environmentally
correlated traits jointly, from complete balanced data to missing
records, unequal designs, sex-limited traits, and international sire
evaluations (MACE):

- **[`ex_06_1`](@ref) (Sections 6.1–6.2)**: Bivariate animal model for
  pre-weaning gain (WWG) and post-weaning gain (PWG) with equal design
  matrices and no missing records, stored in a single `DataFrame`. With
  traits stacked ($y = \mathrm{vec}(Y)$), the MME take the clean
  Kronecker-product form
  $[R_0^{-1}\otimes X'X\ R_0^{-1}\otimes X'Z;\ R_0^{-1}\otimes Z'X\
  R_0^{-1}\otimes Z'Z + G_0^{-1}\otimes A^{-1}]$, which we write out
  with `kron` and check against the general multi-trait assembly
  (`mt_stack`, `mt_blocks`, `mt_rinv`). Multi-trait and univariate EBVs,
  prediction error variances, and reliabilities are placed side by side
  as columns of `ped`, and calf 8's bivariate evaluation is partitioned
  into $\hat{a}_8 = W_1\mathrm{PA}_8 + W_2\mathrm{YD}_8$.
- **[`ex_06_2`](@ref) (Section 6.3)**: Bivariate animal model with equal
  design matrices when calves 4 and 9 are missing PWG. To see how
  missing records enter the MME, we write out the padded Kronecker form
  in which missing observations are set to zero and each calf
  contributes $(R_0[S,S])^{-1}$ on its observed trait subset $S$ (and
  zeros elsewhere), and check it against `mt_stack`, `mt_blocks`, and
  `mt_rinv` (which drop missing records instead of padding). Comparing
  the bivariate and univariate columns of `ped` shows how calves 4 and 9
  gain PWG information from their own WWG records through the genetic
  correlation.
- **[`ex_06_3`](@ref) (Section 6.4)**: Fat yield in parities 1 and 2
  analyzed as two correlated traits with unequal design matrices. We
  take the long-format lactation table from `ex_05_1` and reshape it to
  one row per cow with `unstack`. Because HYS 1–2 occur only in parity 1
  and HYS 3–4 only in parity 2, each trait has its own design matrix
  $X_t$, so the MME replace $I_2 \otimes X$ with
  $X_b = \mathrm{blockdiag}(X_1, X_2)$. We also compute the multi-trait
  $\mathrm{DYD}$ of sire 1 (Section 6.4.2).
- **[`ex_06_4`](@ref) (Section 6.5.1)**: Multivariate analysis with zero
  residual covariance: yearling weight recorded only on males and fat
  yield recorded only on females, in one 17-animal `DataFrame`. Because
  no animal has both traits and $R_0 = \mathrm{diag}(\sigma_{e1}^2,
  \sigma_{e2}^2)$ is diagonal, the fixed-effect and data blocks of the
  two traits are completely separate; we write out the $4 \times 4$
  block MME to show that information flows between males and females
  solely through the off-diagonal blocks $g^{12} A^{-1}$ of
  $G_0^{-1} \otimes A^{-1}$.
- **[`ex_06_5`](@ref) (Section 6.5.2)**: Multi-trait Across-Country
  Evaluation (MACE) of bulls using a sire–maternal-grandsire (sire–MGS)
  pedigree with 5 phantom parent groups. We keep a table of bulls, a
  table of groups, and a table of national proofs (deregressed proofs
  $\mathrm{DRP}$ and effective daughter contributions $\mathrm{EDC}$ in
  each country). Because each bull's residual precision
  $R_{ik}^{-1} = \mathrm{EDC}_{ik}/\sigma_{ei}^2$ is different, we write
  the MME in the general $R^{-1}$ form with explicit stabilization on
  the group diagonals, check it against `mme`, and decompose the MACE
  proofs of bulls 3 and 2 into parent average, national daughter
  deviation, and progeny contribution (Eqns 6.19–6.21).

---

## Chapter 8: Maternal Effects Models

For traits like calf birth weight or weaning weight, a calf's record is
affected both by its own genotype (the direct genetic effect) and by the
milking and mothering ability of its dam (maternal genetic and maternal
permanent environmental effects):

- **[`ex_08_1`](@ref) (Sections 8.1–8.2)**: Maternal animal model for
  calf birth weight on a 14-animal `DataFrame`. Each calf's record is
  linked to its own direct genetic effect ($u$) by $Z$, to its dam's
  maternal genetic effect ($m$) by $W$, and to its dam's permanent
  environment ($pe$) by $S$. Because $u$ and $m$ are correlated across
  all relatives ($\mathrm{var}([u;\ m]) = G_0 \otimes A$), their $2
  \times 2$ block in the MME is coupled by
  $\alpha \otimes A^{-1} = (\sigma_e^2 G_0^{-1}) \otimes A^{-1}$. We
  write out the $4 \times 4$ block MME explicitly, check it against
  `mme`, and add the direct, maternal, and $pe$ solutions to `ped`.
- **[`ex_08_2`](@ref) (Section 8.3)**: Maternal Reduced Animal Model
  (RAM) on the same dataset. Only parents (animals 1–9) have equations
  in the MME; the direct Mendelian sampling of non-parent calves (10–14)
  is absorbed into their residuals. We write the reduced MME in the
  general $R^{-1}$ form, solve for the parents, and then back-solve the
  non-parents in pedigree order—first their direct Mendelian sampling
  (Eqn 8.8), and then their maternal Mendelian sampling by regressing on
  the direct deviation via $g_{12}/g_{11}$ (Eqn 8.10)—recovering the
  exact solutions of the full model in `ex_08_1`.

---

## Chapter 9: Social Interaction Models

When animals are housed in groups (such as pigs in a pen), an
individual's growth rate depends not only on its own direct breeding
value ($\mathrm{DBV}$, $u_D$) but also on the associative (social)
breeding values ($\mathrm{SBV}$, $u_S$) of its pen-mates:

- **[`ex_09_1`](@ref) (Sections 9.2–9.4)**: Social interaction model for
  15 pigs (9 recorded in 3 pens of 3) in one `DataFrame`. We construct
  the associative incidence matrix directly from the pen incidence
  matrix $V$ as $Z_S = (VV' - I)Z_D$ (`VV'` finds records in the same
  pen, and subtracting $I$ removes the pig itself), and write out the
  MME (Eqn 9.6) in the compact form $T'T + \Sigma$ with
  $T = [X\ Z_D\ Z_S\ V\ W]$ (fixed sex, correlated direct and social
  genetic effects, random pen $g$, and random litter $c$). We then fit
  the equivalent correlated-residual model (Eqn 9.7, Section 9.4), which
  omits the explicit pen effect $g$ and instead uses the block-diagonal
  residual covariance $R = \sigma_e^2[(1-\rho)I + \rho VV']$, verifying
  that all fixed, genetic, and litter solutions are identical. Finally,
  we partition pig 7's $(\hat{u}_D, \hat{u}_S)$ into parent average and
  direct/social yield deviations (Section 9.3), and compare total
  breeding values $\mathrm{TBV}_i = u_{D,i} + (n-1)u_{S,i}$ with a
  standard animal model that ignores social effects.

---

## Chapter 10: Random Regression & Covariance Functions

Chapter 10 models longitudinal traits (such as test-day milk or fat
yields recorded along a lactation trajectory) using orthogonal Legendre
polynomials and linear splines:

- **[`ex_10_1`](@ref) (Sections 10.1–10.2)**: Fixed regression test-day
  model. Table 10.1 is entered just as printed in the book (one column
  per cow across 10 Days in Milk, DIM) and reshaped with `stack` to a
  long-format table `rec` with one row per test day. All cows share a
  single fixed lactation curve of order-4 Legendre polynomials ($X_2$),
  while each cow has a single constant daily breeding value ($a_j$) and
  permanent environmental effect ($pe_j$). We write the MME as
  $T'T + \Sigma$ with $T = [X_1\ X_2\ Z\ Q]$, check it against `mme`,
  evaluate the fitted lactation curve, and partition the EBVs by joining
  per-cow yield deviations onto `ped`.
- **[`ex_10_2`](@ref) (Section 10.3)**: Random regression test-day model
  on the same records. Each animal now has its own order-2 Legendre
  genetic trajectory ($u_{j0}, u_{j1}, u_{j2}$) and permanent
  environmental trajectory ($pe_{j0}, pe_{j1}, pe_{j2}$). We build the
  random-regression design matrix
  $Z_{RR} = [\mathrm{Diag}(\phi_0)Z\ \mathrm{Diag}(\phi_1)Z\
  \mathrm{Diag}(\phi_2)Z]$, write the MME as $T'T + \Sigma$ with
  $\sigma_e^2 G^{-1} \otimes A^{-1}$ and $\sigma_e^2 P^{-1} \otimes I$,
  and check it against `mme`. Summing the polynomial covariables from
  day 6 to day 310 gives the weight vector $t$ used to compute 305-day
  breeding values ($\hat{u}t$), their reliabilities
  $1 - (t'C^{ii}t)/(t'Gt)$ (Section 10.3.4), and the multi-trait
  partition of cow 6's random regression coefficients (Section 10.3.2).
- **[`ex_10_3`](@ref) (Sections 10.4 and 10.3.5)**: Covariance functions
  and linear splines. Given a $3 \times 3$ genetic covariance matrix $G$
  at 90, 160, and 240 days of age, the full-order Legendre covariance
  function is simply $\hat{C} = \Phi^{-1}G(\Phi')^{-1}$, written in
  Julia as `Φ \ G / Φ'`. For the reduced-order ($k = 2$) Weighted Least
  Squares (WLS) fit (Section 10.4.1), we write out the vectorized design
  matrix $X = L(\Phi_2 \otimes \Phi_2)D$ (where $L$ extracts the lower
  triangle $\mathrm{vech}(G)$ and $D$ is the duplication matrix of $C$),
  solve $(X'V^{-1}X)^{-1}X'V^{-1}\mathrm{vech}(G)$, and check both fits
  and the $\chi^2$ test against `covariance_function`. We then tabulate
  the continuous variance and covariance curves across ages 90–240 and
  compute the linear spline covariables of Section 10.3.5.

---

## Chapter 11: Genomic Selection & The Bayesian Alphabet

Chapter 11 develops genomic prediction from marker regressions and
SNP-BLUP to GBLUP, ancestral genotype imputation, residual polygenic
effects, haplotype models, the Bayesian alphabet, and multi-trait GBLUP:

- **[`ex_11_1`](@ref) (Sections 11.2–11.3)**: Fitting the first 3 SNPs
  as fixed regressions alongside a random polygenic effect ($u \sim N(0,
  A\sigma_a^2)$), both unweighted and weighted by each bull's Effective
  Daughter Contribution ($\mathrm{EDC}$). Both analyses share the same
  general-form MME
  $[X'R^{-1}X\ X'R^{-1}W;\ W'R^{-1}X\ W'R^{-1}W + A^{-1}/\sigma_a^2]$,
  with $R^{-1} = I/\sigma_e^2$ or
  $\mathrm{diag}(\mathrm{EDC}/\sigma_e^2)$. Direct genomic values
  ($\mathrm{DGV} = Z_3\hat{g}$), polygenic effects ($\hat{u}$), and
  $\mathrm{GEBV} = \mathrm{DGV} + \hat{u}$ are added as columns of the
  `bulls` table.
- **[`ex_11_2`](@ref) (Section 11.4.1)**: SNP-BLUP (ridge regression
  BLUP) fitting all 10 SNPs as random effects with common variance
  $\sigma_g^2 = \sigma_a^2 / 2\sum p_j(1-p_j)$, unweighted and
  EDC-weighted. We write out the general-form MME, check it against
  `mme`, and collect the marker solutions in a `snps` table and the bull
  predictions in `bulls`. Notice that SNPs 7 and 8 are monomorphic in
  the 8 reference bulls: their centred columns are constant (collinear
  with the overall mean), so ridge regression shrinks their estimated
  effects to exactly zero.
- **[`ex_11_3`](@ref) (Sections 11.4.2–11.4.5 and 11.9)**: Genomic BLUP
  (GBLUP) and equivalent formulations. We build VanRaden's first genomic
  relationship matrix $G = ZZ' / 2\sum p_j(1-p_j)$ (and check it against
  `grm`), note that $G$ ($14 \times 14$ from 10 SNPs) has rank at most
  10 so $G + 0.01I$ is inverted, and write out the GBLUP MME
  $[X'X\ X'W;\ W'X\ W'W + \lambda G^{-1}]$. We then show three key
  results:
  1. GBLUP animal solutions $\hat{a}$ agree with $Z\hat{g}$ from
     SNP-BLUP up to the $0.01I$ regularization of $G$;
  2. Individual SNP effects can be back-solved directly from GBLUP as
     $\hat{g} = \frac{1}{k} Z'G^{-1}\hat{a}$ (Example 11.4, checked
     against `snp_from_gblup`);
  3. The selection-index form
     $\hat{a} = G_{\cdot,r}(G_{rr} + \lambda I)^{-1}(y_r -
     \mathbf{1}\hat{\mu})$ (Example 11.5, checked against
     `gblup_index`) needs no $14 \times 14$ inverse of $G$ and
     reproduces the SNP-BLUP DGVs and theoretical reliabilities.
- **[`ex_11_6`](@ref) (Section 11.4.6)**: Base-population allele
  frequencies and ancestral gene-content imputation by the linear method
  of Gengler et al. (2007). Treating each SNP's gene content ($0, 1, 2$)
  as a highly heritable trait ($\lambda = 0.01$) on the pedigree gives
  the same left-hand side $[X'X\ X'W;\ W'X\ W'W + \lambda A^{-1}]$ for
  every marker, so we solve for all 10 SNPs at once using a matrix
  right-hand side $[X'M;\ W'M]$ and compare with
  `base_allele_frequencies` and the GLS estimator.
- **[`ex_11_7`](@ref) (Section 11.5)**: Genomic evaluation with a
  residual polygenic effect ($u \sim N(0, A\sigma_u^2)$ accounting for
  $10\%$ of $\sigma_a^2$). We write both the SNP-BLUP + polygenic model
  (Eqn 11.14, $T = [X\ W\ Z_r]$) and the GBLUP + polygenic model (Eqn
  11.16, $T = [X\ W\ W]$) in the compact form $T'T + \Sigma$, check them
  against `mme`, and compare their polygenic, genomic, and total
  breeding values in the `bulls` table.
- **[`ex_11_8`](@ref) (Section 11.6)**: Haplotype-based GBLUP. Phased
  haplotypes across 9 SNPs are divided into 3 blocks of 3 SNPs; we write
  out the pseudo-SNP dosage coding (counting how many strands of each
  animal carry each distinct 3-SNP haplotype allele in a block) and
  check it against `pseudo_snps`. From the centred pseudo-SNPs we form
  $G_h = Z_h Z_h' / 2\sum pq$, write out the GBLUP MME
  $[n\ \mathbf{1}';\ \mathbf{1}\ I + \lambda K^{-1}]$, and compare the
  SNP-based and haplotype-based DGVs (both with the book's misprinted
  row for animal 5 and with the corrected haplotype coding).
- **[`ex_11_9`](@ref) (Section 11.7)**: The Bayesian Alphabet
  (**BayesA**, **BayesB**, **BayesC**, and **BayesC$\pi$**) fitted by
  single-site Gibbs sampling with residual updating. We verify the
  deterministic starting values and residuals, run MCMC chains for all
  four priors, and collect the posterior mean SNP effects and inclusion
  probabilities alongside the book's values in a `snps` table and the
  candidate DGVs in `bulls`. We also highlight what happens to SNP 7,
  which is monomorphic ($z_7'z_7 = 0$) in the reference bulls: because
  the likelihood carries no information about $g_7$, its inclusion
  indicator is drawn from the prior with probability $1 - \pi$.
- **[`ex_11_13`](@ref) (Section 11.8)**: Multivariate GBLUP for WWG and
  PWG of the 5 recorded calves from Example 6.1. Because every genotyped
  calf has its own record ($Z = I$), the MME are the Kronecker equations
  of `ex_06_1` with $Z = I$ and $G^{-1}$ in place of $A^{-1}$. The
  resulting bivariate DGVs are compared side by side with the
  pedigree-based EBVs from `ex_06_1` in a `calves` table.

---

## Chapter 12: Single-Step GBLUP and APY

Chapter 12 combines genotyped and ungenotyped animals in a single
evaluation (ssGBLUP) and shows how to invert $G$ in linear time for
large genotyped populations (APY):

- **[`ex_12_1`](@ref) (Section 12.1)**: Single-step GBLUP (ssGBLUP). We
  compute $A_{22}$ for the 14 genotyped bulls by Colleau's algorithm,
  blend $G_b = 0.95G + 0.05A_{22}$, and solve the $2 \times 2$ linear
  system that matches the mean diagonal and off-diagonal elements of
  $G_t = a + bG_b$ to $A_{22}$ (checked against `tune_grm`). We then
  build the single-step inverse
  ```math
  H^{-1} = A^{-1} + \begin{bmatrix} 0 & 0 \\ 0 & G_t^{-1} - A_{22}^{-1} \end{bmatrix}
  ```
  explicitly by adding $G_t^{-1} - A_{22}^{-1}$ to the genotyped block
  of $A^{-1}$ (checked against `hinv`), solve the ssGBLUP MME
  $[X'X\ X'W;\ W'X\ W'W + \lambda H^{-1}]$, and compare the ssGBLUP
  solutions with pedigree BLUP across all 26 animals in `ped`—showing
  how paternal half-sib candidates 22–26, which receive identical
  pedigree BLUPs ($0.030$), are differentiated by their marker
  genotypes in ssGBLUP.
- **[`ex_12_2`](@ref) (Section 12.4)**: Algorithm for Proven and Young
  (APY) sparse inversion of $G$, using bulls 15, 20, 22, and 26 as the
  core subset. We write out the conditional variance of each non-core
  bull given the core bulls,
  $m_{ii} = g_{ii} - g_{ic}G_{cc}^{-1}g_{ci}$, assemble
  $G_{\mathrm{APY}}^{-1}$ from its core and non-core blocks (checking it
  against `apy_ginv`), and compare the resulting GBLUP DGVs with those
  from the full inverse in the `bulls` table.

---

## Chapter 13: Non-Additive Genetic Models

Chapter 13 extends the linear mixed model to capture dominance
deviations (within-locus interactions) and epistasis (between-locus
interactions), using both pedigrees and genome-wide markers:

- **[`ex_13_1`](@ref) (Sections 13.2–13.3.1)**: Pedigree-based additive
  and dominance model for 12 pigs in one `DataFrame`. We write out
  Cockerham's rule for the dominance relationship matrix,
  $D_{ij} = \frac{1}{4}(A_{s_i s_j}A_{d_i d_j} + A_{s_i d_j}A_{d_i
  s_j})$, check it against `drm`, and form the MME as $T'T + \Sigma$
  with $T = [X\ Z\ Z]$ (since the same incidence matrix $Z$ links a
  pig's record to its additive value $a$ and dominance deviation $d$).
- **[`ex_13_2`](@ref) (Section 13.3.2)**: Solving directly for total
  genetic merit $g = a + d$ on the same data. Instead of two random
  effects per animal, we fit a single genetic vector $g$ with
  covariance $V = A\sigma_a^2 + D\sigma_d^2$, solve the smaller MME
  $[X'X\ X'Z;\ Z'X\ Z'Z + \sigma_e^2 V^{-1}]$, and recover the additive
  and dominance components afterwards as $\hat{a} = \sigma_a^2 A V^{-1}
  \hat{g}$ and $\hat{d} = \sigma_d^2 D V^{-1}\hat{g}$—confirming that
  they match the separate model of `ex_13_1` to machine precision.
- **[`ex_13_3`](@ref) (Section 13.4)**: Genomic additive and dominance
  models (Example 13.3) and genomic inbreeding depression (Example
  13.4) for 15 genotyped pigs. We write out both the centred additive
  coding $Z_a = M - 2p'$ and the orthogonal dominance coding $H$ of
  Vitezica et al. (2013) ($-2q^2, 2pq, -2p^2$ for genotypes $2, 1, 0$),
  verify $G = Z_a Z_a'/2\sum pq$ and $D = HH'/\sum(2pq)^2$ against
  `grm`, and fit the MME $T'T + \Sigma$ ($T = [X\ Z\ Z]$) without and
  with the proportion of homozygous loci $f_i$ as a fixed covariate for
  inbreeding depression.
- **[`ex_13_5`](@ref) (Section 13.5)**: Genomic model adding additive
  $\times$ additive epistasis ($aa_g$). We write out the epistatic
  relationship matrix $G_{AA}$ as the elementwise (Hadamard) product
  `G .* G` scaled by its mean diagonal (checked against
  `epistatic_grm`), assemble the MME $T'T + \Sigma$ with $T = [X\ Z\ Z\
  Z]$, and collect the additive, dominance, epistatic, and total genetic
  values in the `pigs` table.

---

## Chapter 14: Crossbreeding & Breed of Origin of Alleles

Chapter 14 addresses multibreed populations and terminal crossbreeding
programs where alleles trace back to distinct founder breeds with
different genetic variances:

- **[`ex_14_1`](@ref) (Sections 14.2–14.4)**: Multibreed animal models
  for 11 animals from two breeds and their crosses, kept in an `animals`
  table. We compute breed fractions ($f_1, f_2$) and breed segregation
  coefficients ($h_{12}$), build the partial relationship matrices
  $A_1, A_2, A_{12}$ (García-Cortés & Toro, 2006), and write out the MME
  for three models as $T'T + \Sigma$:
  1. the breed-specific model fitting $a^{(1)}, a^{(2)}, a^{(12)}$
     separately;
  2. the equivalent combined model fitting total merit $a = a^{(1)} +
     a^{(2)} + a^{(12)}$ with $G = \sigma_1^2 A_1 + \sigma_2^2 A_2 +
     \sigma_{12}^2 A_{12}$;
  3. the random-regression approximation of Strandén & Mäntysaari (2013)
     (Example 14.2), which uses the single full-pedigree matrix $A$
     weighted by $\sqrt{f_1}, \sqrt{f_2}, \sqrt{h_{12}}$.
- **[`ex_14_3`](@ref) (Section 14.7)**: Breed of Origin of Alleles (BOA)
  models for purebred and crossbred performance (Christensen et al.,
  2014), first using pedigree partial relationships $A^{(1)}, A^{(2)}$
  (Example 14.3) and then using breed-specific genomic matrices
  $G^{(1)}, G^{(2)}$ built from phased crossbred alleles (Example 14.4).
  Because crossbred records have a different residual variance
  ($\sigma_x^2 = 4.75$) from purebred records ($\sigma_e^2 = 4.0$), the
  MME are written out in the general $R^{-1}$ form with a $2 \times 2$
  purebred–crossbred genetic block $S_p^{-1} \otimes K_p^{-1}$ for each
  breed.

---

## Chapter 15: Non-Linear & Threshold Models

Categorical traits (such as calving ease scored in ordered classes)
violate the normality and linearity assumptions of standard BLUP and are
modeled via an underlying continuous liability:

- **[`ex_15_1`](@ref) (Sections 15.1–15.2)**: Threshold (probit) sire
  model for calving ease in 3 ordered categories (Gianola & Foulley,
  1983). The data are entered as a table of 20 herd–sex–sire subclasses
  with category counts (`n1`, `n2`, `n3`). After estimating the
  thresholds ($t_1, t_2$), fixed effects, and sire transmitting
  abilities on the liability scale by Fisher scoring
  (`threshold_model`), we write out the category probabilities $P(Y = k
  \mid \eta) = \Phi(t_k - \eta) - \Phi(t_{k-1} - \eta)$ (checked against
  `category_probabilities`). For the linear-model comparison, we expand
  the subclass counts to one row per calving using `stack` and collect
  all sire solutions and probabilities in a `sires` table.
- **[`ex_15_2`](@ref) (Section 15.3)**: Joint analysis of a continuous
  trait (birth weight, BW) and a binary threshold trait (calving
  difficulty, CD) across 47 calvings (Foulley et al., 1983), stored in a
  table of calvings. Conditioning the liability of CD on observed BW
  introduces the residual regression $\beta = \sigma_{e_{12}} /
  \sigma_{e_1}^2$. We compare the joint threshold-linear estimates and
  sire-specific difficult-calving probabilities with the bivariate
  linear approximation, whose MME are written out in the Kronecker form
  of `ex_06_1` and checked against `mme`.

---

## Chapter 17: Variance Component Estimation

Before breeding values can be predicted, the genetic and residual
variance components must be estimated from the data:

- **[`ex_17_1`](@ref) (Sections 17.3 and 17.5)**: Variance components of
  a sire model by Henderson's Method 3 and by fitting canonical sums of
  squares. We write out Method 3 directly from the quadratic-form
  reductions $R(b) = y'X(X'X)^-X'y$ and $R(b, s)$ and the expected sire
  coefficient $\mathrm{tr}(Z'S_X Z)$ (checked against `henderson3`). We
  then form the orthogonal sire contrasts, fit their mean squares by
  Ordinary Least Squares (`C \ ms`) and by a 3-line iterated Generalized
  Least Squares (GLS) loop (checked against `fit_mean_squares`), and
  compare all three sets of estimates in a table.
- **[`ex_17_2`](@ref) (Section 17.7)**: Restricted Maximum Likelihood
  (REML) for the univariate animal model by Average Information
  (AI-REML) and Expectation–Maximization (EM-REML), on a single
  8-animal `DataFrame`. Using the MME at the starting values, we write
  out both the restricted log-likelihood
  $\log L_R = -\frac{1}{2}(\log|C| + \log|A| + N\log\sigma_e^2 +
  q\log\sigma_a^2 + y'Py)$ and the first EM-REML update
  $\sigma_a^{2(1)} = (\hat{a}'A^{-1}\hat{a} + \mathrm{tr}(A^{-1}C^{aa})
  \sigma_e^2)/q$ (Eqns 17.5–17.6), check them against `reml`, and add
  the EBVs at the converged AI-REML estimates to `ped`.

---

## Chapter 18: Bayesian Analysis & Gibbs Sampling

Chapter 18 introduces Markov Chain Monte Carlo (MCMC) inference for
location parameters and variance components in univariate and
multivariate animal models:

- **[`ex_18_1`](@ref) (Section 18.2)**: Univariate animal model Gibbs
  sampler on the data table of `ex_04_1`. We write out the first Gibbs
  sweep as an explicit loop of single-site normal updates,
  $\theta_i \sim N((r_i - \sum_{j \ne i} C_{ij}\theta_j)/C_{ii},\;
  \sigma_e^2/C_{ii})$, replaying the textbook's 10 normal deviates to
  reproduce Round 1 exactly and checking it against `gibbs_sweep!`. We
  then run full chains—first holding variances fixed to confirm that the
  posterior means converge to the BLUP solutions, and then sampling
  $(\sigma_e^2, \sigma_a^2)$ under proper scaled inverse-$\chi^2$
  priors—comparing all three sets of animal estimates in `ped`.
- **[`ex_18_2`](@ref) (Section 18.3)**: Multivariate block Gibbs sampler
  with inverse-Wishart updates of $R_0$ and $G_0$ on the bivariate data
  table of `ex_06_1`. We verify that the posterior means with fixed
  (co)variances match multi-trait BLUP, demonstrate how a missing PWG
  record on calf 4 is imputed at each sweep by drawing from its
  conditional normal distribution given WWG, and tabulate the prior and
  posterior means of $R_0$ and $G_0$ when (co)variances are sampled.

---

## Chapter 19: Iterative Solvers for Large-Scale Systems

When an evaluation involves millions of animals, inverting or even
storing the MME coefficient matrix $C$ is impractical. Chapter 19
demonstrates stationary iterative methods, Iteration on Data (IOD), and
Preconditioned Conjugate Gradients (PCG):

- **[`ex_19_1`](@ref) (Section 19.4)**: Jacobi (Example 19.1, with
  relaxation $\omega = 0.8$ on animal equations) and Gauss–Seidel
  (Example 19.2) iteration on the explicit MME of `ex_04_1`. Mrode's
  starting values (sex means for fixed effects, deviations from the sex
  mean for recorded calves, and 0 for unrecorded ancestors) are computed
  from the data table with `groupby`. Both iterative schemes are written
  out as explicit loops—showing the difference between using the
  previous round's vector $x^{(k)}$ in Jacobi versus immediately using
  updated entries in Gauss–Seidel—checked against `jacobi` and
  `gauss_seidel`, and their convergence criteria are tabulated side by
  side across all 20 rounds.
- **[`ex_19_3`](@ref) (Sections 19.5.1, 19.5.2, and 19.6)**: Iteration
  on Data (IOD) for the animal model without assembling $C$:
  - Gauss–Seidel on data (`AnimalModelIOD`) without unknown parent
    groups (Example 19.3) and with unknown parent groups 9 and 10
    (Example 19.4, both unconstrained and with group 9 fixed to 0 to
    recover the QP solution of `ex_04_4`);
  - Preconditioned Conjugate Gradient (PCG) on data (Example 19.6) with
    diagonal preconditioner $M = \mathrm{diag}(C)$, written out as an
    explicit loop over the matrix-free product $q = Cp$ (`iod(q, p)`)
    and checked against `pcg`. The solutions of Gauss–Seidel IOD, PCG,
    and the direct solution of the MME are compared in a table.
- **[`ex_19_5`](@ref) (Section 19.5.3)**: Maternal Reduced Animal Model
  (the system of `ex_08_2`) solved by Gauss–Seidel iteration without
  constraining a fixed-effect level. Because the herd and pen equations
  are linearly dependent, the coefficient matrix is singular and
  Gauss–Seidel converges to one particular generalized-inverse solution.
  Comparing this unconstrained solution with the constrained solution of
  `ex_08_2` ($\mathrm{herd}_1 = 0$) illustrates a fundamental property
  of linear models: the individual herd and pen numbers differ, but the
  estimable cell means $\mathrm{herd}_i + \mathrm{pen}_j$ and all random
  genetic and permanent environmental predictions agree.
