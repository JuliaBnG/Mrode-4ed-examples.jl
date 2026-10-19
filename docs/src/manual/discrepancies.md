# Textbook Discrepancies & Errata

When working through a textbook with a calculator or writing your own
code, few things are more frustrating than getting a slightly different
number and not knowing whether the mistake is in your algebra or in a
printed table.

During the implementation and verification of the worked examples in
Mrode & Pocrnic (2023), several typographical slips, sign omissions, and
intermediate-rounding differences were found in the 4th edition text. In
each case, the example script explains the cause of the difference in a
comment and checks the mathematically verified value. All of these
notes are gathered below by chapter for easy reference.

---

## Chapter 6: Multivariate Models

- **Example 6.3 (`ex_06_3`) — Daughter Yield Deviation for Sire 1**:
  In Section 6.4.2, the book computes $\mathrm{DYD}$ for sire 1 using a
  first-parity yield deviation for daughter 4 of
  $\mathrm{YD}_{4,1} = 25.7$. However, from the printed record
  $y_{4,1} = 201$ and HYS solution $\hat{b}_1 = 175.7$, the subtraction
  is $201 - 175.7 = 25.3$. Correcting that arithmetic slip gives
  $\mathrm{DYD}_1 = [24.254, 32.154]'$ (instead of the printed
  $[24.521, 32.154]'$) and $M_3\mathrm{DYD}_1 = [7.388, 7.358]'$
  (instead of $[7.439, 7.387]'$).
- **Example 6.4 (`ex_06_4`) — Zero Residual Covariance**:
  In the printed solutions table, animal 6 is listed as
  $(-5.012, -2.098)$, an exact duplicate of animal 5. In the pedigree,
  however, dam 6 is the mother of male 11, who has a high yearling
  weight record ($300\text{ kg}$ in HYS 2 vs $250\text{ kg}$ for dam 5's
  son, animal 10). Solving the MME gives $+5.012$ and $+2.098$ for dam
  6.
- **Example 6.5 (`ex_06_5`) — Multi-Trait Across-Country Evaluation
  (MACE)**:
  1. For bull 3, who has a record only in country 1, the weight matrix
     $W_1$ is printed with a non-zero $(1,2)$ entry of $-0.3614$.
     Because bull 3 has no progeny, $W_1 + W_2 = I_2$, and since column
     2 of $W_2 = [0.5771\ 0;\ 0.3614\ 0]$ is zero, the $(1,2)$ entry of
     $W_1$ is $0$.
  2. Equation (6.21) for predicting a bull's proof in a country where he
     has no daughters is printed with a minus sign; because
     $g^{21}/g^{22} = -g_{12}/g_{11}$, the regression term requires a
     plus sign, which also reproduces the book's own value of $5.001$
     for bull 3 in country 2.
  3. For bull 2, the text computes the progeny contribution through his
     maternal grandson (bull 3) as $4\hat{a}_3 - 2\hat{a}_7$, omitting
     the maternal granddam group ($\mathrm{G5}$) of bull 3; the exact
     progeny contribution is $4\hat{a}_3 - 2\hat{a}_7 - \hat{g}_5$.

---

## Chapter 10: Random Regression Models

- **Example 10.1 (`ex_10_1`) — Legendre Fixed Lactation Curve**:
  From Days in Milk (DIM) 174 onward, the values printed for the fixed
  lactation curve ($11.0407, 10.9156, \dots$) do not equal
  $\Phi\hat{b}_2$ evaluated from the book's own fixed-regression
  solutions $\hat{b}_2$ and Legendre matrix $\Phi$ (at DIM 174,
  $\Phi_{174}\hat{b}_2 = 11.0994$).
- **Example 10.2 (`ex_10_2`) — 305-Day Weights and Trajectory
  Variance**:
  1. The cumulative 305-day polynomial weight vector $t =
     \sum_{d=6}^{310} \phi(d)$ in the book sums Legendre polynomials
     computed from 4-decimal rounded coefficients, giving $t_3 =
     -1.5561$ instead of $-1.5451$ at full precision.
  2. In Section 10.3.4, the permanent environmental variance on a
     305-day basis, $k'Pk$ (with $k = t$), is printed as $323462.969$,
     whereas multiplying the book's own $t$ and $P$ gives $320126$.
- **Section 10.3.5 (`ex_10_3`) — Linear Spline Covariables**:
  In the table of linear spline covariables with knots at days 4, 106,
  208, and 310, the printed weights between two knots place the larger
  weight on the farther knot; `ex_10_3` computes standard linear
  interpolation (where a test day closer to knot $k$ than to knot $k+1$
  receives the larger weight on knot $k$).

---

## Chapter 11: Genomic Selection & Bayesian Methods

- **Example 11.1 (`ex_11_1`) — Candidate IDs**:
  In the table of results, the 6 selection candidates are labelled
  25–30 rather than 21–26 (in the 26-animal pedigree of Table 11.1,
  ancestors are 1–12, reference bulls are 13–20, and candidates are
  21–26).
- **Example 11.6 (`ex_11_6`) — GLS Base Allele Frequency**:
  For SNP 1, the text reports a Generalized Least Squares (GLS) base
  mean gene content $\hat{\mu} = 2\hat{p}$ of $0.575$. Evaluating
  $(\mathbf{1}'K\mathbf{1})^{-1}\mathbf{1}'K M_{:,1}$ gives $0.590$ with
  $K = A_{22}^{-1}$ (close to Gengler's mixed-model estimate of $0.597$)
  and $0.526$ with the submatrix $K = (A^{-1})_{22}$.
- **Example 11.7 (`ex_11_7`) — Residual Polygenic Effect**:
  In Table 11.4, the columns headed "DGV" actually tabulate the total
  breeding value $\mathrm{GEBV} = \mathrm{DGV} + \hat{u}$ (genomic plus
  residual polygenic effect). In addition, the polygenic component was
  computed using the 5-generation relationship matrix $A$ printed in
  Example 11.3 rather than the 2-generation pedigree of Table 11.1.
- **Example 11.8 (`ex_11_8`) — Haplotype Pseudo-SNPs**:
  1. For animal 5 in block 2 (SNPs 4–6), the phased haplotypes are `001`
     and `000` (alleles 2 and 5 of that block), so the pseudo-SNP row
     is `01001`, but the book prints `01000`. The book's $2\sum pq =
     4.42$, $G_h$, and haplotype GBLUP solutions follow from that
     misprinted row; `ex_11_8` verifies both the book's table and the
     corrected haplotype matrix ($2\sum pq = 4.56$).
  2. The text states a correlation of $0.999$ between the SNP-based and
     haplotype-based $\mathrm{DGV}$s, whereas the correlation computed
     from the book's own printed solution vectors is $0.992$.
- **Example 11.9 (`ex_11_9`) — BayesA Starting Residual**:
  At the deterministic starting values ($\mu^{(0)} = 9.888$ and
  $g_j^{(0)} = 0.05$), the first bull's residual is
  $9.0 - 9.888 - 9(0.05) = -1.338$, whereas the text prints $-1.388$.
- **Examples 11.10 & 11.12 (`ex_11_9`) — BayesB Variances & Monomorphic
  SNP 7**: In BayesB (Example 11.10), the book's posterior SNP variances
  ($\approx 0.30$) are larger than the stationary MCMC posterior mean
  ($\approx 0.12$ on average across loci, stable over seeds and chain
  lengths). All other posterior means—including the BayesC$\pi$
  posterior $\pi = 0.51$ in Example 11.12—are reproduced. (Note that SNP
  7 is $0$ in all 8 reference bulls, so $z_7'z_7 = 0$ and its inclusion
  indicator is sampled from the prior with probability $1 - \pi$; an
  earlier discrepancy in BayesC$\pi$ was caused by a $0/0$ log-odds
  evaluation for monomorphic markers in `Breeding` prior to v0.1.6.)

---

## Chapter 12: Single-Step GBLUP and APY

- **Example 12.2 (`ex_12_2`) — APY Inverse Ordering and Sign Slip**:
  The matrix $G_{\mathrm{APY}}^{-1}$ is printed in the textbook with the
  4 core bulls (15, 20, 22, 26) ordered first, followed by the 10
  non-core bulls, rather than in original bull ID order (13–26). In the
  solutions table, the minus sign on bull 17's $\mathrm{DGV}$ is
  omitted ($0.471$ printed instead of $-0.471$).

---

## Chapter 14: Crossbreeding Models

- **Example 14.2 (`ex_14_1`) — Random-Regression Multibreed Table**:
  Table 14.3 reports the weighted breed contributions
  $\sqrt{f_p}\,\hat{u}_p$ and $\sqrt{h_{12}}\,\hat{u}_{12}$ (which sum
  directly to the total breeding value $\hat{a}^*$) rather than the raw
  random-regression solutions $\hat{u}_p$.
- **Example 14.3 (`ex_14_3`) — Pedigree Breed of Origin**:
  The solutions match Table 14.5 within $0.005$, but the book's printed
  values for exchangeable founders (such as animals 1 and 2, which have
  symmetric roles in the design) show a slight rounding asymmetry
  ($-0.060$ and $+0.055$ vs exact $\pm 0.057$).
- **Example 14.4 (`ex_14_3`) — Genomic Breed of Origin**:
  The matrix $G^{(1)}$ printed in the text does not match Equation
  (14.25) (for instance, its $(1,1)$ entry is printed as $1.815$ instead
  of $0.607$), but the final BOA solutions in Table 14.9 *were* computed
  from the correct $G^{(1)}$ of Equation (14.25) and are reproduced
  here.

---

## Chapter 15: Non-Linear & Threshold Models

- **Example 15.1 (`ex_15_1`) — Threshold Sire Model**:
  In Table 15.3, the linear-model comparison column cannot be reproduced
  from the category counts in Table 15.2. In the table of category
  probabilities by sire, sire 4's probabilities differ by $0.003$.

---

## Chapter 17: Variance Component Estimation

- **Section 17.5 (`ex_17_1`) — Canonical Sums of Squares (OLS & SEs)**:
  Fitting the three canonical mean squares ($0.3025, 0.1250, 0.1800$) on
  $C = [1\ 1.5;\ 1\ 1.0;\ 1\ 0.0]$ by unweighted Ordinary Least Squares
  gives $\hat{\sigma}_e^2 = 0.151$ and $\hat{\sigma}_s^2 = 0.062$
  (whereas the text prints $0.143$ and $0.079$). The iterated GLS
  estimates ($0.163$ and $0.047$) match the book, while the asymptotic
  sampling variances $\mathrm{diag}((C'V^{-1}C)^{-1})$ are
  $(0.051, 0.062)$ with standard errors $(0.226, 0.250)$ (compared with
  $0.216$ and $0.234$ in the text).
- **Section 17.7 (`ex_17_2`) — REML Starting Solutions**:
  At the starting variance components ($\sigma_e^2 = 0.4, \sigma_a^2 =
  0.2$), calf 4's breeding value is printed as $\hat{a}_4 = -0.254$; its
  sign is actually $+0.254$, which is required to obtain the book's own
  residual for calf 4 ($2.6 - 2.144 - 0.254 = 0.2022$).

---

## Chapter 18: Bayesian Analysis & Gibbs Sampling

- **Example 18.1 (`ex_18_1`) — Improper Flat Priors on Variance
  Components**: The textbook specifies flat priors ($\nu_e = \nu_a =
  -2$, $S^2 = 0$) for the variance components. With only 5 phenotypic
  records and 8 animals, a flat prior on $\sigma_a^2$ yields an improper
  joint posterior that diverges at $\sigma_a^2 \to 0$, causing a long
  Gibbs chain to be absorbed at zero genetic variance. After replaying
  Round 1 exactly as in the book, `ex_18_1` uses weakly informative
  proper scaled inverse-$\chi^2$ priors ($\nu = 4$) for the full chain.

---

## Chapter 19: Iterative Solvers

- **Example 19.3 (`ex_19_3`) — Gauss–Seidel Iteration on Data**:
  The column headed "At convergence" in the book is actually round 20;
  the stated convergence criterion ($< 10^{-7}$) is already satisfied at
  round 9.
- **Example 19.4 (`ex_19_3`) — Iteration on Data with Unknown Parent
  Groups**: The column of constrained solutions ($5.474, \dots, -1.795$)
  printed in the book has not yet converged to the exact constrained
  solution of Example 4.4 ($5.458, \dots, -1.769$).
- **Example 19.5 (`ex_19_5`) — Maternal RAM by Gauss–Seidel**:
  The dataset is that of Table 8.1 (Examples 8.1 and 8.2), where the
  second fixed factor is called "Pen" (levels 1 and 2); in Example 19.5
  the book labels those same two levels "Sex" (Male and Female). In
  addition, because Mrode's stopping rule divides the sum of squared
  changes by the sum of squared solutions ($\approx 30^2$ for the herd
  means), stopping at $10^{-14}$ still leaves differences of order
  $10^{-5}$ in the random effects; `ex_19_5` iterates to $10^{-24}$ so
  that the random effects and estimable cell means match Example 8.2 to
  8 decimal places.
