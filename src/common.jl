# Helpers shared by all examples.

const NCHECKS = Ref(0)
const FAILED = String[]

"""
    check(label, got, ref; atol = 1e-3) -> Bool

Compare `got` with reference values (usually as printed, i.e. rounded,
in the book) and print a one-line verdict with the largest absolute
difference. Every comparison is counted, and failures are recorded for
[`run_examples`](@ref).
"""
function check(label, got, ref; atol = 1e-3)
    d = maximum(abs.(collect(Iterators.flatten(got)) .- collect(Iterators.flatten(ref))))
    ok = d ≤ atol
    @printf("  %-48s %s (max |Δ| = %.2e)\n", label, ok ? "ok" : "DIFFERS", d)
    NCHECKS[] += 1
    ok || push!(FAILED, label)
    ok
end

"""
    show_table(; digits = 3, cols...)

Print named columns as a rounded table.
"""
show_table(; digits = 3, cols...) =
    println(DataFrame([k => (eltype(v) <: AbstractFloat ? round.(v; digits) : v)
                       for (k, v) in cols]))

"""
    list_examples([io])

Print the catalogue of examples: function, chapter and title.
"""
function list_examples(io::IO = stdout)
    for e in EXAMPLES
        @printf(io, "%-10s ch %2d  %s\n", e.name, e.chapter, e.title)
    end
end

_select(::Colon) = EXAMPLES
_select(ch::Integer) = filter(e -> e.chapter == ch, EXAMPLES)
_select(chs::AbstractVector{<:Integer}) = filter(e -> e.chapter in chs, EXAMPLES)
_select(name::Symbol) = filter(e -> e.name == name, EXAMPLES)
_select(names::AbstractVector{Symbol}) = filter(e -> e.name in names, EXAMPLES)

"""
    run_examples(selection = :; quiet = false) -> NamedTuple

Run examples and summarise their comparisons with the book. `selection`
is `:` (all), a chapter number or vector of chapters, or an example name
(e.g. `:ex_11_2`) or vector of names. With `quiet = true` the output of
the examples is suppressed. Returns, per example, the number of checks
and the labels of those that failed, plus the totals.
"""
function run_examples(selection = :; quiet = false)
    sel = _select(selection)
    isempty(sel) && throw(ArgumentError("no example matches $selection"))
    results = map(sel) do e
        n0, f0 = NCHECKS[], length(FAILED)
        quiet || println("\n", "="^72, "\n", e.name, " – ", e.title)
        f = getfield(@__MODULE__, e.name)
        quiet ? redirect_stdout(f, devnull) : f()
        (name = e.name, checks = NCHECKS[] - n0, failed = FAILED[f0+1:end])
    end
    nfail = sum(length(r.failed) for r in results)
    total = sum(r.checks for r in results)
    @printf("\n%d examples, %d checks, %d failed\n", length(results), total, nfail)
    for r in results, l in r.failed
        println("  ", r.name, ": ", l)
    end
    (examples = results, checks = total, failed = nfail)
end
