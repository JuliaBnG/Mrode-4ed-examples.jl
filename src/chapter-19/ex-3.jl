"""
    ex_19_3() -> Nothing

Run Examples 19.3, 19.4, and 19.6 from Mrode & Pocrnic (2023): Iteration on Data (IOD)
for the animal model:
1. **Gauss–Seidel IOD** (Example 19.3)
2. **Gauss–Seidel IOD with unknown parent groups** (Example 19.4)
3. **Preconditioned Conjugate Gradient (PCG) on data** (Example 19.6)

# Iteration on Data (IOD)
In large-scale genetic evaluations, forming and storing the MME coefficient matrix ``C``
is computationally prohibitive. IOD computes updates directly by scanning data records
and pedigree links:
-  **Data scan**: Accumulates adjusted phenotypic contributions
  ``\\sum (y_i - \\hat{y}_{i,-j})`` for fixed effects and recorded animals.
- **Pedigree scan**: Accumulates relationship contributions from parents, mates, and progeny
  using the rules of ``A^{-1}``.

# Unknown Parent Groups (Example 19.4)
Incorporates phantom parent groups 9 and 10. The unconstrained system is singular
(due to collinearity between group effects and base population mean). Setting Group 9 to
zero yields the exact QP-transformed solutions of Example 4.4.

# Preconditioned Conjugate Gradient (Example 19.6)
Solves ``C\\mathbf{x} = \\mathbf{r}`` matrix-free via the action ``C \\mathbf{d}`` on data:
```math
\\alpha_k = \\frac{\\mathbf{r}_k' M^{-1} \\mathbf{r}_k}{\\mathbf{p}_k' C \\mathbf{p}_k},
    \\quad
\\mathbf{x}_{k+1} = \\mathbf{x}_k + \\alpha_k \\mathbf{p}_k
```
with diagonal preconditioner ``M = \\mathrm{diag}(C)``.

# Returns
Nothing: the iterates and solutions are printed and checked against the book.
"""
function ex_19_3()
    (; ped, calf, sex, y, m, x0, α) = data_19()

    # Step 1: Gauss–Seidel iteration on data (IOD)
    println("Example 19.3: Gauss–Seidel iteration on data")
    iod = AnimalModelIOD(y, [sex], calf, ped.sire, ped.dam, α)
    x, it, conv, H = gauss_seidel(iod; x0, tol = 0, maxiter = 20, history = true)
    check(
        "round 1",
        H[:, 2],
        [4.333, 3.400, 0.333, -0.083, -0.021, -0.119, -0.376, 0.392, -0.364, 0.282],
    )
    check(
        "round 20",
        x,
        [4.359, 3.404, 0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.249, 0.183],
    )
    check(
        "= iteration on the MME (Example 19.2)",
        H,
        gauss_seidel(m.lhs, m.rhs; x0, tol = 0, maxiter = 20, history = true)[4];
        atol = 1e-12,
    )
    println(
        "  criterion < 1e-7 reached after ",
        gauss_seidel(iod; x0, tol = 1e-7)[2],
        " rounds",
    )
    v = similar(m.rhs)
    iod(v, x0)
    check("matrix-free C·x = MME C·x", v, m.lhs * x0; atol = 1e-12)

    # Step 2: IOD with unknown parent groups (UPG 9 and 10)
    println("Example 19.4: … with unknown parent groups 9 and 10")
    sg = [9, 9, 9, 1, 3, 1, 4, 3]
    dg = [10, 10, 10, 10, 2, 2, 5, 6]
    iodg = AnimalModelIOD(y, [sex], calf, sg, dg, α)
    xg0 = [x0; 0; 0]
    x, it, conv, H = gauss_seidel(iodg; x0 = xg0, tol = 1e-7, history = true)
    check(
        "round 1",
        H[:, 2],
        [
            4.333,
            3.400,
            0.333,
            -0.083,
            -0.021,
            -0.119,
            -0.376,
            0.392,
            -0.364,
            0.282,
            0.153,
            -0.176,
        ],
    )
    show_table(effect = ["male", "female", string.(1:10)...], unconstrained = x)
    println(
        "  (the unconstrained system is singular; the book stops at 1e-7 with",
        " male 4.509, G9 0.949, G10 −0.820)",
    )

    # Group 9 fixed at zero: the solutions of Example 4.4 (QP-transformed MME)
    xf, it =
        gauss_seidel(iodg; x0 = xg0, fix = Dict(11 => 0.0), tol = 1e-14, maxiter = 100_000)
    check(
        "G9 = 0: solutions of Example 4.4",
        xf,
        [
            5.458,
            4.313,
            -0.767,
            -0.923,
            -0.963,
            -1.268,
            -1.099,
            -0.728,
            -1.338,
            -0.768,
            0,
            -1.769,
        ],
    )
    println(
        "  (the book's constrained column, 5.474 … −1.795, is not the converged",
        " solution of Example 4.4)",
    )
    check(
        "linear differences agree (male − female)",
        x[1] - x[2],
        xf[1] - xf[2];
        atol = 0.01,
    )

    # Step 3: Preconditioned conjugate gradient (PCG) on data
    println("Example 19.6: preconditioned conjugate gradient on data")
    D = diag(iod)
    check(
        "diagonal preconditioner M",
        D,
        [3, 2, 3.667, 4, 4, 4.667, 6, 6, 5, 5];
        atol = 1e-3,
    )
    check("right-hand side", rhs(iod), [13.0, 6.8, 0, 0, 0, 4.5, 2.9, 3.9, 3.5, 5.0])
    x, it, conv, H =
        pcg(iod, rhs(iod); M = D, tol = 1e-6, criterion = :change, history = true)
    check("round 1", H[:, 2], [3.430, 2.691, 0, 0, 0, 0.763, 0.383, 0.514, 0.554, 0.791])
    check(
        "round 3",
        H[:, 4],
        [3.835, 3.122, 0.475, 0.224, 0.272, 0.390, 0.249, 0.547, 0.193, 0.537],
    )
    check(
        "round 5",
        H[:, 6],
        [4.280, 3.154, 0.170, 0.116, 0.058, 0.032, -0.072, 0.435, -0.178, 0.334],
    )
    check(
        "round 10",
        H[:, 11],
        [4.359, 3.404, 0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.249, 0.183],
    )
    println("  converged after ", it, " rounds (book: 10)")
    nothing
end
