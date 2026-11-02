using Documenter
using Mrode4edExamples

# JuliaBnG Sepia and Black themes, added to Documenter's theme picker.
# The files in src/assets/juliabng/ are built by docs-themes/build.jl in the
# juliabng.github.io repository; rebuild there rather than editing them here.
const JULIABNG_ASSETS = [
    [asset("assets/juliabng/$t.css"; class = :css, islocal = true,
           attributes = Dict(Symbol("data-theme-name") => t))
     for t in ("juliabng-sepia", "juliabng-black")];
    "assets/juliabng/themes.js"
]

makedocs(
    sitename = "Mrode4edExamples.jl",
    authors = "Xijiang Yu",
    modules = [Mrode4edExamples],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://juliabng.github.io/Mrode-4ed-examples/",
        size_threshold_warn = 200 * 2^10,
        size_threshold = 300 * 2^10,
        assets = JULIABNG_ASSETS,
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Overview & Quick Start" => "manual/overview.md",
            "Catalogue of Chapters" => "manual/chapters.md",
            "Textbook Discrepancies" => "manual/discrepancies.md",
        ],
        "API Reference" => "api.md",
    ],
)

if !isempty(get(ENV, "DOCUMENTER_KEY", ""))
    deploydocs(
        repo = "github.com/JuliaBnG/Mrode-4ed-examples.jl.git",
        deploy_repo = "github.com/JuliaBnG/juliabng.github.io.git",
        dirname = "Mrode-4ed-examples",
        devbranch = "main",
        branch = "gh-pages",
        forcepush = true,
    )
elseif get(ENV, "GITHUB_ACTIONS", "") == "true"
    @warn "Skipping documentation deployment because DOCUMENTER_KEY is not configured."
end
