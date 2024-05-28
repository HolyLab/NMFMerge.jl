using NMFMerge
using Documenter

DocMeta.setdocmeta!(NMFMerge, :DocTestSetup, :(using NMFMerge); recursive=true)

makedocs(;
    modules=[NMFMerge],
    authors="youdongguo <1010705897@qq.com> and contributors",
    sitename="NMFMerge.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/NMFMerge.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/NMFMerge.jl",
    devbranch="main",
)
