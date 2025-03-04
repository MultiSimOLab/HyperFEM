using HyperFEM
using Documenter

DocMeta.setdocmeta!(HyperFEM, :DocTestSetup, :(using HyperFEM); recursive=true)

makedocs(;
    modules=[HyperFEM],
    authors="MultiSimo_Group",
    repo="https://github.com/jmartfrut/Mimosa.jl/blob/{commit}{path}#{line}",
    sitename="HyperFEM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jmartfrut.github.io/HyperFEM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jmartfrut/HyperFEM.jl",
    devbranch="main",
)
