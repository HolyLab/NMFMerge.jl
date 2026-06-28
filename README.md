# NMFMerge

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/NMFMerge.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/NMFMerge.jl/dev/)
[![Build Status](https://github.com/HolyLab/NMFMerge.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/NMFMerge.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/NMFMerge.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/NMFMerge.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

NMFMerge improves nonnegative matrix factorization (NMF) by *over-factorizing*
and then merging components in pairs. It projects NMF solutions from a
higher-dimensional space to a lower-dimensional one by optimally and sequentially
merging NMF component pairs, which tends to reach better and more reproducible
optima than factorizing directly into the target number of components.

This approach is motivated by the idea that convergence of NMF becomes poor when
one is forced to make difficult tradeoffs in describing different features of the
data matrix; performing an initial factorization with an excessive number of
components grants the opportunity to escape such constraints and reliably
describe the full behavior of the data matrix. Later, any redundant or noisy
components are identified and merged together.

The package implements the technique in [An optimal pairwise merge algorithm
improves the quality and consistency of nonnegative matrix
factorization](https://ieeexplore.ieee.org/abstract/document/11071940) (Guo &
Holy, IEEE Trans. Signal Process. 2025, doi:10.1109/TSP.2025.3585893).

The concept of NMF-Merge in an illustrative example:
![Sample Figure](images/ovrsimumerge.png)

The data matrix is $\mathbf{X}=\mathbf{WH}+\mathbf{N}$, where $\mathbf{W}$ and $\mathbf{H}$ are rank-5 and $\mathbf{N}$ denotes the noise. The 'Good' and 'Bad' factorizations (blue box) represent two local minima reached by rank-5 NMF across 1000 different initializations, while the factorizations in the green boxes denote minima identified by NMF with $r\ge 6$. (These minima were found from every random initialization tested, but uniqueness is not required.) The higher-rank solutions can be merged to produce a rank 5 factorization using NMFMerge. Thus, by first identifying a higher-rank NMF, and then merging to lower rank, you can more reliably identify high-quality solutions. We demonstrate this experimentally in the linked manuscript.

## Installation

NMFMerge is a registered package. From the Julia REPL, type `]` to enter package
mode and run:

```julia
pkg> add NMFMerge
```

## Usage

Build your nonnegative data matrix `X` and call `nmfmerge`:

```julia
using NMFMerge

result = nmfmerge(X, 5 => 4)   # factorize with 5 components, merge down to 4
W, H = result.W, result.H
```

`ncomponents` may be a pair `n1 => n2` (overcomplete count `=>` final count) or a
single integer final count (in which case `nmfmerge` defaults to an approximate
20% component excess before merging).

The exported building blocks — `colnormalize`, `merge_pq`, and `merge_replay` —
let you drive merging yourself, for example when choosing the number of
components by merging all the way down and locating a "knee" in the merge-error
sequence.

See the [documentation](https://HolyLab.github.io/NMFMerge.jl/stable/) for a
full worked example, the lower-level workflow, and the complete API reference.

## Citation

Thanks for citing this work! See the "Cite this repository" link in the "About"
bar for format options.
