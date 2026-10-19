# Overview & Quick Start

## Learning Linear Mixed Models by Building Them

When you first study *Linear Models for the Prediction of the Genetic
Merit of Animals* (Mrode & Pocrnic, 2023), the hardest step is often
bridging the gap between the matrix equations on the page and the code
that actually computes the breeding values. If a program hides the
incidence matrices $X$ and $Z$, the inverse relationship matrix
$A^{-1}$, and the Mixed Model Equations (MME) inside a high-level
formula interface, it is difficult to see how each record and pedigree
link contributes to the system.

`Mrode4edExamples.jl` is written so that you can read each Julia script
side by side with the textbook and see every matrix, table, and equation
explicitly:

1. **Data and solutions live in `DataFrame`s**:
   In most examples, the pedigree and phenotypic records are placed in a
   single `DataFrame` with one row per animal. Animal IDs are simply the
   row numbers `1:n`, and base animals (founders) without records have
   `missing` in the trait column:
   - `ainv(ped)` reads only the `sire` and `dam` columns to build
     $A^{-1}$ by Henderson's rules.
   - `rec = dropmissing(ped, :wwg)` selects the recorded animals to form
     the observation vector $y$ and incidence matrices $X$ and $Z$.
   - As the example runs, animal-level results—EBVs, prediction error
     variances, reliabilities, parent averages ($\mathrm{PA}$), yield
     deviations ($\mathrm{YD}$), progeny contributions ($\mathrm{PC}$),
     or genomic predictions—are added directly as new columns of `ped`.
   When an animal has repeated records (such as multiple lactations in
   Chapter 5 or test-day yields in Chapter 10), the pedigree and the
   records are kept in two tables and combined with `groupby`,
   `combine`, `leftjoin!`, `stack`, or `unstack`.

2. **Two forms of the Mixed Model Equations**:
   For the linear mixed model $y = Xb + Za + e$ with
   $\mathrm{var}(a) = G = A\sigma_a^2$ and $\mathrm{var}(e) = R$, the
   general MME are
   ```math
   \begin{bmatrix}
   X'R^{-1}X & X'R^{-1}Z \\
   Z'R^{-1}X & Z'R^{-1}Z + A^{-1}/\sigma_a^2
   \end{bmatrix}
   \begin{bmatrix} \hat{b} \\ \hat{a} \end{bmatrix}
   =
   \begin{bmatrix} X'R^{-1}y \\ Z'R^{-1}y \end{bmatrix}.
   ```
   When every record has the same residual variance, $R = I\sigma_e^2$,
   multiplying both sides by $\sigma_e^2$ gives the textbook's
   variance-ratio form with $\lambda = \sigma_e^2 / \sigma_a^2$:
   ```math
   \begin{bmatrix}
   X'X & X'Z \\
   Z'X & Z'Z + \lambda A^{-1}
   \end{bmatrix}
   \begin{bmatrix} \hat{b} \\ \hat{a} \end{bmatrix}
   =
   \begin{bmatrix} X'y \\ Z'y \end{bmatrix}.
   ```
   Notice how the two forms relate:
   - In the general $R^{-1}$ form, the inverse of the coefficient matrix
     gives the prediction error variance directly,
     $\mathrm{PEV}(\hat{a}_i) = (C^{-1})_{aa,ii}$, and heterogeneous
     residuals (such as Mendelian-sampling weights in a Reduced Animal
     Model, EDC weights in MACE and SNP-BLUP, or crossbred residuals in
     BOA) are handled naturally through $R^{-1}$.
   - In the $\lambda$ form, the coefficient matrix is the general one
     multiplied by $\sigma_e^2$ (its inverse is scaled by
     $1/\sigma_e^2$), so $\mathrm{PEV}(\hat{a}_i) = d_i \sigma_e^2$ and
     $r_i^2 = 1 - d_i \lambda$, where $d_i$ is the $i$-th diagonal of
     the animal block of $C^{-1}$.

   In [`ex_04_1`](@ref), both forms are written out as Julia block
   matrices and shown to give the same solutions. Throughout the
   package, each example writes the MME in whichever form fits the model
   best—a $2 \times 2$ block matrix, a Kronecker product with `kron` for
   multi-trait models, or $T'T + \Sigma$ with $T = [X\ Z\ W\ \dots]$
   when several random effects are fitted—and then checks that the
   explicit matrix equals `mme(...)`.

3. **Algorithms written out in plain Julia**:
   Whenever a new concept is introduced—back-solving non-parents in RAM,
   building $A^{*-1}$ for unknown parent groups from $A^{-1}$ and $Q$,
   Cockerham's rule for dominance $D$, Vitezica's genomic dominance
   coding, pseudo-SNPs from haplotypes, GBLUP back-solving and selection
   index, tuning $G$ to $A_{22}$ and forming $H^{-1}$, the APY inverse
   from core and non-core blocks, WLS covariance functions, Henderson's
   Method 3, REML updates, a Gibbs sweep, Jacobi, Gauss–Seidel, or
   matrix-free PCG—the script writes the calculation out in a few lines
   of Julia and compares it with the library routine.

---

## Naming Convention & Return Values

Every worked example is a function `ex_CC_N()` in
`src/chapter-CC/ex-N.jl`, where `CC` is the two-digit chapter number and
`N` is the first example number covered by that script:

| Function | Chapter | Topic |
|:---|:---:|:---|
| `ex_04_1()` | 4 | Univariate animal model, accuracy, and PA/YD/PC partition |
| `ex_04_2()` | 4 | Sire model compared with $\frac{1}{2}\hat{a}$ from the animal model |
| `ex_04_3()` | 4 | Reduced Animal Model (RAM) and back-solving non-parents |
| `ex_04_4()` | 4 | Unknown parent groups via the Quaas–Pollak (QP) transformation |
| `ex_05_1()` | 5 | Repeatability model and daughter yield deviations (DYD) |
| `ex_06_1()` | 6 | Bivariate animal model (Kronecker MME, reliabilities, partition) |
| `ex_08_1()` | 8 | Maternal animal model (correlated direct and maternal effects) |
| `ex_09_1()` | 9 | Social interaction model (DBV, SBV, TBV, correlated residuals) |
| `ex_10_2()` | 10 | Random regression test-day model with Legendre polynomials |
| `ex_11_2()` | 11 | SNP-BLUP (unweighted and EDC-weighted) |
| `ex_11_3()` | 11 | GBLUP, back-solving SNP effects, and selection-index form |
| `ex_11_9()` | 11 | BayesA, BayesB, BayesC, and BayesC$\pi$ via residual updating |
| `ex_12_1()` | 12 | Single-step GBLUP (blending, tuning $G$ to $A_{22}$, and $H^{-1}$) |
| `ex_12_2()` | 12 | Algorithm for Proven and Young (APY) sparse $G^{-1}$ |
| `ex_13_1()` | 13 | Pedigree additive and dominance model (Cockerham's rule) |
| `ex_14_1()` | 14 | Multibreed partial relationship matrices and RR approximation |
| `ex_15_1()` | 15 | Threshold (probit) sire model for ordered categorical traits |
| `ex_17_2()` | 17 | AI-REML and EM-REML for the univariate animal model |
| `ex_18_1()` | 18 | Gibbs sampling for the univariate animal model |
| `ex_19_3()` | 19 | Iteration on data (Gauss–Seidel, UPGs, and matrix-free PCG) |

Most functions return a `NamedTuple` containing the `DataFrame` of
animals or records (with the solutions attached as columns) along with
the solution vectors (`b`, `a`, `u`, etc.) and intermediate matrices so
you can inspect or plot them in the REPL.

---

## Running an Example and Reading the Checks

Each example uses `check(label, got, ref; atol = 1e-3)` to compare
computed quantities against either an equivalent formulation (at machine
precision, `atol = 1e-10` or `1e-12`) or the rounded values printed in
the textbook (typically `atol = 1e-3`).

```julia
using Mrode4edExamples

res = ex_04_1()
```

Running `ex_04_1()` prints the input table, the MME coefficient matrix
and right-hand side, each numerical check, and the completed table:

```text
Example 4.1: animal model, pre-weaning gain of beef calves
8×5 DataFrame
 Row │ calf   sex      sire   dam    wwg
     │ Int64  String?  Int64  Int64  Float64?
─────┼─────────────────────────────────────────
   1 │     1  missing      0      0  missing
   2 │     2  missing      0      0  missing
   3 │     3  missing      0      0  missing
   4 │     4  M            1      0        4.5
   5 │     5  F            3      2        2.9
   6 │     6  F            1      2        3.9
   7 │     7  M            4      5        3.5
   8 │     8  M            3      6        5.0
...
  general R⁻¹ form = λ form                        ok (max |Δ| = 2.66e-15)
  mme(; σ²e) LHS = [X'X X'Z; Z'X Z'Z+λA⁻¹]         ok (max |Δ| = 0.00e+00)
  mme(; Rinv) LHS = general form                   ok (max |Δ| = 1.39e-17)
...
  sex effects                                      ok (max |Δ| = 5.02e-04)
  breeding values                                  ok (max |Δ| = 4.59e-04)
  diagonals of C⁻¹                                 ok (max |Δ| = 4.59e-04)
  reliabilities                                    ok (max |Δ| = 9.18e-04)
  SEP                                              ok (max |Δ| = 1.98e-03)
  PEV of general form = dᵢ σ²e                     ok (max |Δ| = 7.11e-15)
  reliability() without inverting C                ok (max |Δ| = 6.66e-16)
...
Data with solutions:
8×12 DataFrame
 Row │ calf   sex      sire   dam    wwg        ebv      d        r2       sep      PA       YD           PC
     │ Int64  String?  Int64  Int64  Float64?   Float64  Float64  Float64  Float64  Float64  Float64?     Float64?
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     1  missing      0      0  missing      0.098    0.471    0.058    4.341    0.0    missing            0.217
   2 │     2  missing      0      0  missing     -0.019    0.492    0.016    4.437    0.0    missing           -0.038
   3 │     3  missing      0      0  missing     -0.041    0.456    0.087    4.273    0.0    missing           -0.082
   4 │     4  M            1      0        4.5   -0.009    0.428    0.145    4.136    0.049        0.141       -0.313
   5 │     5  F            3      2        2.9   -0.186    0.428    0.144    4.138   -0.03        -0.504       -0.49
   6 │     6  F            1      2        3.9    0.177    0.442    0.115    4.206    0.04         0.496        0.406
   7 │     7  M            4      5        3.5   -0.249    0.442    0.116    4.204   -0.097       -0.859  missing
   8 │     8  M            3      6        5.0    0.183    0.422    0.155    4.11     0.068        0.641  missing
```

Where a printed number in the book differs from the exact result of its
own formula (for instance, a sign slip or intermediate rounding), the
script notes the discrepancy in a comment, checks the corrected value,
and documents the details in
[Textbook Discrepancies](discrepancies.md).

---

## Exploring the Catalogue & Running Batches

Use [`list_examples`](@ref) to print all 42 examples, or
[`run_examples`](@ref) to run a chapter, a list of examples, or the
entire suite:

```julia
# Print the catalogue of all examples
list_examples()

# Run all examples in Chapter 11
run_examples(11)

# Run specific examples by name
run_examples([:ex_12_1, :ex_12_2])

# Run all 42 examples quietly and inspect the summary
results = run_examples(; quiet = true)
println("Total checks: ", results.checks, ", Failed: ", results.failed)
```
