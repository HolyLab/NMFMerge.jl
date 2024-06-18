# NMFMerge

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/NMFMerge.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/NMFMerge.jl/dev/)
[![Build Status](https://github.com/HolyLab/NMFMerge.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/NMFMerge.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/NMFMerge.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/NMFMerge.jl)

This package includes the code of the paper 'An optimal pairwise merge algorithm improves the quality and consistency of nonnegative matrix factorization`.
It is used for merging components in non-negative matrix factorization.

Suppose you have the NMF solution ``W`` and ``H`` with ``r`` componenent, **colmerge2to1pq** function can merge ``r`` components to ``n``components. The details of this function is:

**colmerge2to1pq**(W, H, n)

This function merges components in ``W`` and ``H`` (columns in ``W`` and rows in ``H``) from original number of components to ``n`` components (``n``columns and rows left in ``W`` and ``H`` respectively).

To use this function:
`Wmerge, Hmerge, mergeseq = colmerge2to1pq(W, H, n)`, where ``Wmerge`` and ``Hmerge`` are the merged results with ``n`` components. ``mergeseq`` is the sequence of merge pair ids ``(id1, id2)``, which is the components id of single merge.

Before merging components, the columns in ``W`` are required to be normalized to 1. The normalization can be realized by **colnormalize** function or anyother method you like.

**colnormalize**(W, H, p)

This function normalize ``||W[:, i]||_p = 1`` for ``i in 1:size(W, 2)``. For this paper ``p=2``

To use this function:
`Wnormalized, Hnormalized = colnormalize(W, H, p)`

If you already have a merge sequence and want to merge from ``size(W, 2)`` components to ``n`` components, you can use the function:
**mergecolumns**(W, H, mergeseq; tracemerge)
keyword argurment ``tracemerge``: save ``Wmerge`` and ``Hmerge`` at each merge stage if ``tracemerge=true``. default ``tracemerge=false``.

To use this function:
`Wmerge, Hmerge, WHstage, Err = mergecolumns(W, H, mergeseq; tracemerge)`, where ``Wmerge`` and ``Hmerge`` are the merged results. ``WHstage::Vector{Tuple{Matrix, Matrix}}`` includes the results of each merge stage. ``WHstage=[]`` if ``tracemerge=false``. ``Err::Vector`` includes merge penalty of each merge stage.

Demo:
Prerequisite: NMF.jl
Considering the ground truth

```math
\begin{align} 
        \begin{aligned}
        \label{simu_matrix}
            \mathbf{W} = \begin{pmatrix}
                6 & 0 & 4 & 9 \\
                0 & 4 & 8 & 3 \\
                4 & 4 & 0 & 7 \\
                9 & 1 & 1 & 1 \\
                0 & 3 & 0 & 4 \\
                8 & 1 & 4 & 0 \\
                0 & 0 & 4 & 2 \\
                0 & 9 & 5 & 5 
            \end{pmatrix}, \quad
            \mathbf{H}^{\mathrm{T}} = \begin{pmatrix}
                6 & 0 & 3 & 4 \\
                10 & 10 & 5 & 9 \\
                8 & 2 & 0 & 10 \\
                2 & 9 & 2 & 7 \\
                0 & 10 & 4 & 7 \\
                1 & 6 & 0 & 0 \\
                2 & 0 & 0 & 0 \\
                10 & 0 & 8 & 0
            \end{pmatrix}
        \end{aligned}
    \end{align}
```

```julia
julia> X = W*H
8×8 Matrix{Int64}:
 84  161  138   83   79   6  12   92
 36  107   38   73   93  24   0   64
 52  143  110   93   89  28   8   40
 61  114   84   36   21  15  18   98
 16   66   46   55   58  18   0    0
 60  110   66   33   26  14  16  112
 20   38   20   22   30   0   0   32
 35  160   68  126  145  54   0   40
```
Running NMF (HALS algorithm) on $\mathbf{X}$ with NNDSVD initialization

```julia
julia> f = svd(X);
julia> result_hals = nnmf(float(X), 4; init=:nndsvd, alg=:cd, initdata=f, maxiter = 10^6, tol = 1e-4);
julia> result_hals.objvalue/sum(abs2, X)
0.00019519131697246967
```

Running NMF Merge on $\mathbf{X}$ with NNDSVD initialization
```julia
julia> f = svd(X);
julia> result_over = nnmf(float(X), 5; init=:nndsvd, alg=:cd, initdata=f, maxiter = 10^6, tol = 1e-4);
julia> W1, H1 = result_over.W, result_over.H;
julia> W1normed, H1normed = colnormalize(W1, H1);
julia> Wmerge, Hmerge, mergesq = colmerge2to1pq(copy(W1normed), copy(H1normed), 4);
julia> result_renmf = nnmf(float(X), 4; init=:custom, alg = :cd, maxiter=10^6, tol=1e-4, W0=copy(Wmerge), H0=copy(Hmerge));
julia> result_renmf.objvalue/sum(abs2, X)
8.873476732142566e-7
```

## Citation
The code is welcomed to be used in your publication, please cite:


