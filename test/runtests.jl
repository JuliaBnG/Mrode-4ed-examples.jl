using Mrode4edExamples
using Test

# Every example compares its results with the values printed in the book;
# an example passes when all its comparisons do.
@testset "Mrode & Pocrnic (2023) examples" begin
    for e in EXAMPLES
        @testset "$(e.name)" begin
            r = run_examples(e.name; quiet = true)
            @test r.checks > 0
            @test r.failed == 0
        end
    end
end
