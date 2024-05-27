using NMF_Merge
using Documenter

DocMeta.setdocmeta!(NMF_Merge, :DocTestSetup, :(using NMF_Merge); recursive=true)

makedocs(;
    modules=[NMF_Merge],
    authors="youdongguo <1010705897@qq.com> and contributors",
    sitename="NMF_Merge.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/NMF_Merge.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/NMF_Merge.jl",
    devbranch="main",
)
