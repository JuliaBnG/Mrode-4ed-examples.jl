"""
    ex_19_3() -> DataFrame

Examples 19.3, 19.4 & 19.6: Iteration on Data (IOD) and Preconditioned Conjugate Gradient
(PCG) for the animal model (Mrode & Pocrnic, 2023, Sections 19.5.1, 19.5.2, and 19.6;
Schaeffer & Kennedy, 1986; Strandén & Lidauer, 1999).

# Model Formulation
In large national evaluations, assembling and storing the sparse coefficient matrix ``C`` is
avoided by computing the action of ``C`` directly from a scan of the data and pedigree files
(`AnimalModelIOD`):
- **Data scan**: Accumulates phenotype and incidence contributions for fixed effects and
  recorded animals.
- **Pedigree scan**: Accumulates ``\\lambda A^{-1}`` contributions from each animal's
  parents, mates, and progeny using Henderson's rules.

Three matrix-free solvers are demonstrated on the data of [`ex_04_1`](@ref) and
[`ex_04_4`](@ref):
1. **Gauss–Seidel IOD** (Example 19.3): Produces the exact same sequence of iterates as
   Gauss–Seidel on the assembled matrix ``C`` (Example 19.2).
2. **Gauss–Seidel IOD with unknown parent groups** (Example 19.4): Replaces unknown sires
   and dams by phantom groups 9 and 10; constraining group 9 to zero recovers the
   QP-transformed solution of [`ex_04_4`](@ref).
3. **Preconditioned Conjugate Gradient (PCG) on data** (Example 19.6): Solves ``Cx = b``
   using only the matrix-free product ``q_k = C p_k`` (`iod(q, p)`) and the diagonal
   preconditioner ``M = \\mathrm{diag}(C)``.

# Key Concepts Illustrated
1. **Equivalence of IOD and assembled-matrix iteration**: Every round of `gauss_seidel(iod)`
   matches `gauss_seidel(m.lhs, m.rhs)` to machine precision, and `iod(v, x0)` equals
   `m.lhs * x0`.
2. **Matrix-free PCG written out as an explicit loop**: We write out the PCG recurrence
   (step length ``\\alpha_k = (r_k' z_k)/(p_k' C p_k)``, conjugate direction update
   ``\\beta_k = (r_{k+1}' z_{k+1})/(r_k' z_k)``) directly around `iod(q, pp)`, check it
   against `pcg`, and compare the Gauss–Seidel IOD, PCG, and direct MME solutions in a
   table.

# Notes on Text Discrepancies
1. In Example 19.3, the column headed "At convergence" is round 20, whereas the stated
   ``10^{-7}`` criterion is already met at round 9.
2. In Example 19.4, the constrained column printed in the book (`5.474 … -1.795`) has not
   yet converged to the exact constrained solution of Example 4.4 (`5.458 … -1.769`).

# Returns
A `DataFrame` comparing the solutions (`effect`, `gauss_seidel_iod`, `pcg`, and `direct`).
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
    #    unknown sires → group 9, unknown dams → group 10 (equations 11 and 12)
    sg = replace(ped.sire, 0 => 9)
    dg = replace(ped.dam, 0 => 10)
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

    #    PCG written out on the matrix-free product C·p = iod(q, p), with the
    #    diagonal preconditioner M: from x = 0, r = b, z = M⁻¹r, p = z, each round
    #    α = r'z / p'Cp, x += αp, r -= αCp, z = M⁻¹r, β = r'z (new) / r'z (old), p = z + βp
    b = rhs(iod)
    xp = zeros(length(b))
    rp = copy(b)
    zp = rp ./ D
    pp = copy(zp)
    q = similar(b)
    for _ = 1:it
        iod(q, pp)
        αk = (rp'zp) / (pp'q)
        xp += αk * pp
        rz = rp'zp
        rp -= αk * q
        zp = rp ./ D
        pp = zp + (rp'zp) / rz * pp
    end
    check("PCG written out = pcg() (last round)", xp, x; atol = 1e-10)

    sol = DataFrame(
        effect = ["male"; "female"; "animal " .* string.(1:8)],
        gauss_seidel_iod = gauss_seidel(iod; x0, tol = 1e-12)[1],
        pcg = x,
        direct = solve_mme(m),
    )
    println(
        "\nSolutions: Gauss–Seidel on data, PCG on data and direct solution of the MME:",
    )
    show_table(sol)
    sol
end
