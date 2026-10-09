# Textbook Discrepancies & Errata

During the reimplementation and numeric verification of the worked examples in Mrode & Pocrnic (2023), several typographical slips, arithmetic discrepancies, and rounding differences were identified in the printed text. 

To ensure reproducibility, `Mrode4edExamples.jl` checks computed solutions against the mathematically verified values and documents the exact nature of each deviation below.

---

## Chapter 6: Multivariate Models

- **Example 6.3 (Daughter Yield Deviation for Sire 1)**:
  The book computes $\mathrm{DYD}$ for sire 1 using an individual yield deviation $\mathrm{YD}_{41} = 25.7$. However, the actual subtraction $y_{41} - \hat{b}_1 = 201 - 175.7 = 25.3$. Using the correct yield deviation yields the exact mathematical result.
- **Example 6.4 (Zero Residual Covariance)**:
  In the printed solutions table, animal 6 is printed as a duplicate copy of animal 5 with incorrect signs. The Julia solution calculates the true solutions from the MME.
- **Example 6.5 (MACE)**:
  - The matrix $W_1$ for bull 3 contains a spurious non-zero $(1,2)$ element in the printed table.
  - Equation (6.21) contains a typographical minus sign where a plus sign is algebraically required.
  - The progeny contribution ($\mathrm{PC}$) of bull 2 omits the maternal grandsire (MGD) group of the grandson.

---

## Chapter 10: Random Regression Models

- **Example 10.1 (Ali-Schaeffer Lactation Curve)**:
  The lactation curve values printed from Days in Milk (DIM) 174 onward do not correspond to $\Phi \hat{b}_2$ as defined by the regression covariates.
- **Example 10.2 (Legendre Polynomials & Trajectory Variance)**:
  The scalar $k'Pk$ is printed as $323463$, whereas evaluation with the book's own trajectory vector $t$ and projection matrix $P$ yields $320126$. The text uses 4-decimal rounded Legendre coefficients in intermediate steps.

---

## Chapter 11: Genomic Selection & Bayesian Methods

- **Examples 11.1–11.2 (Marker Regression)**:
  Selection candidates are labelled 25–30 in the text. For bull 17, the SNP-BLUP direct genomic value (DGV) is printed without its negative sign (printed as positive 12.1 instead of $-12.1$).
- **Example 11.6 (Base Population Allele Frequencies)**:
  The GLS base allele frequency of $0.575$ printed in the text cannot be reproduced from the stated GLS equation ($0.526$ unweighted or $0.590$ weighted).
- **Example 11.7 (Polygenic Effect Blending)**:
  The columns labelled "DGV" in Table 11.4 actually represent $\mathrm{DGV} + \text{polygenic effect}$, and the polygenic effects utilize the printed five-generation numerator relationship matrix $A$.
- **Example 11.8 (Haplotype Models)**:
  In animal 5's pseudo-SNP genotype string (block 2), a typographical substitution occurred in the book. As a result, $G_h$, $2\sum p q = 4.42$, and the haplotype GBLUP values in the text follow from the typo. The printed correlation of $0.999$ between SNP and haplotype evaluations is actually $0.992$ when evaluated from the book's own published solution vector.
- **Example 11.9 (BayesA)**:
  The first-round residual estimate $\hat{e}_1$ is $-1.338$, whereas the text prints $-1.388$.
- **Examples 11.10 & 11.12 (BayesB & BayesC$\pi$)**:
  In BayesB, the printed posterior SNP variances ($\approx 0.30$) differ slightly from the MCMC stationary distribution ($\approx 0.12 - 0.25$). In BayesC$\pi$, the printed posterior inclusion $\pi = 0.51$ differs from the empirical posterior mean ($\approx 0.67$), which is stable across random number seeds. All other posterior means and GEBVs match closely.

---

## Chapter 12: Single-Step GBLUP and APY

- **Example 12.2 (Algorithm for Proven and Young)**:
  The matrix $G^{-1}_{\mathrm{APY}}$ in the textbook is printed in APY reordered sequence (core animals first, followed by non-core animals) rather than original pedigree identification order. Bull 17's DGV is printed without its negative sign.

---

## Chapter 14: Crossbreeding Models

- **Example 14.2 (Reduced-Rank Multi-Breed Approximation)**:
  Table 14.3 lists $\sqrt{f_p} u_p$ rather than the raw breed solutions $u_p$.
- **Example 14.3 (Breed of Origin - Pedigree)**:
  Printed solutions match within $0.005$, but founder solutions are not perfectly symmetric for exchangeable founder lineages.
- **Example 14.4 (Breed of Origin - Genomic)**:
  The printed genomic matrix $G^{(1)}$ does not directly follow from Equation (14.25), although the final solutions in Table 14.9 do.

---

## Chapter 15: Non-Linear & Threshold Models

- **Example 15.1 (Threshold Model)**:
  The linear-model comparison column cannot be reproduced directly from Table 15.2. Sire 4's predicted categorical probabilities differ by approximately $0.003$.
